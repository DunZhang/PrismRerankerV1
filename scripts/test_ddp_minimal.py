"""最小化 DDP 多卡训练测试脚本.

用途：排查 pytorch + transformers + DDP 在不同机器上能否跑通.

启动方式：
    torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py
    # 或指定卡：
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py

排查思路：
    1. 看初始化阶段：env 变量、device 绑定、NCCL 后端是否能起来
    2. 看通信阶段：all_reduce 是否能跑通（NCCL 死锁通常卡在这里）
    3. 看模型阶段：DDP 包装 + 一次前向反向是否正常
    4. 任一阶段卡住或报错，就能定位到具体环节
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def log(msg: str, rank: int | None = None) -> None:
    prefix = f"[rank {rank}]" if rank is not None else "[main]"
    print(f"{prefix} {msg}", flush=True)


def print_env(rank: int) -> None:
    keys = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
    ]
    info = {k: os.environ.get(k, "<unset>") for k in keys}
    log(f"env: {info}", rank)


def step1_init() -> tuple[int, int, int, torch.device]:
    """Step 1: 初始化进程组并绑定设备."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    log(
        f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
        f"device_count={torch.cuda.device_count()}",
        rank,
    )
    print_env(rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法测试 DDP")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log(f"set device -> {device}", rank)

    log("init_process_group(backend=nccl) ...", rank)
    t0 = time.time()
    dist.init_process_group(backend="nccl", init_method="env://")
    log(f"init_process_group done in {time.time() - t0:.2f}s", rank)

    return rank, local_rank, world_size, device


def step2_allreduce(rank: int, world_size: int, device: torch.device) -> None:
    """Step 2: 测试 NCCL all_reduce 通信."""
    log("all_reduce test ...", rank)
    x = torch.tensor([float(rank + 1)], device=device)
    t0 = time.time()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    expected = sum(range(1, world_size + 1))
    log(
        f"all_reduce done in {time.time() - t0:.2f}s, "
        f"got={x.item()}, expected={expected}",
        rank,
    )
    assert int(x.item()) == expected, "all_reduce 结果错误"


def step3_ddp_train(rank: int, local_rank: int, device: torch.device) -> None:
    """Step 3: DDP 包装 + 一次前向反向."""
    log("build model & DDP wrap ...", rank)
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    log("run 3 training steps ...", rank)
    for step in range(3):
        x = torch.randn(8, 128, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        optimizer.zero_grad()
        out = ddp_model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        log(f"step {step} loss={loss.item():.4f}", rank)


def step4_transformers(rank: int, local_rank: int, device: torch.device) -> None:
    """Step 4 (可选): 用一个极小的 transformers 模型走一遍 DDP."""
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        log("transformers 未安装，跳过 step4", rank)
        return

    log("build tiny transformers model ...", rank)
    config = AutoConfig.for_model(
        "bert",
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    # add_pooling_layer=False: 避免 pooler 参数因 loss 只用 last_hidden_state 而拿不到 grad
    model = AutoModel.from_config(config, add_pooling_layer=False).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

    log("run 2 transformers steps ...", rank)
    for step in range(2):
        input_ids = torch.randint(0, 128, (4, 16), device=device)
        out = ddp_model(input_ids=input_ids)
        loss = out.last_hidden_state.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        log(f"hf step {step} loss={loss.item():.6f}", rank)


def main() -> None:
    if "RANK" not in os.environ:
        print(
            "请用 torchrun 启动，例如：\n"
            "  torchrun --nproc_per_node=2 scripts/test_ddp_minimal.py",
            file=sys.stderr,
        )
        sys.exit(1)

    rank, local_rank, world_size, device = step1_init()
    try:
        step2_allreduce(rank, world_size, device)
        step3_ddp_train(rank, local_rank, device)
        step4_transformers(rank, local_rank, device)
        dist.barrier()
        log("ALL PASSED ✅", rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
