"""Minimal usage example for the Prism GGUF server.

Prerequisites:
    bash web_application/start_server.sh

Usage:
    uv run python web_application/prism_gguf_example.py
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from shared.prompts import (
    TRAINING_INSTRUCTION,
    TRAINING_SYSTEM_PROMPT,
    render_raw_prompt,
)

SERVER_URL = "http://localhost:54580/v1"
MODEL_NAME = "prism-gguf"

query="""
大陆地区，4090和5090哪个性价比高，自己深度学习用
"""
document="""
![](//i.sso.sina.com.cn/images/login/thumb_default.png)
# 显卡界「王炸对决」!RTX 5090 vs RTX 4090,你的需求决定谁更值得入手?
![](//k.sinaimg.cn/n/sinakd20260418s/285/w1080h805/20260418/bb72-167ba7c3fcf38328378d1f5e2bf5bafe.jpg/w700d1q75cms.jpg)
英伟达（深圳）售后服务
RTX 5090  VS 4090
Strong confrontation
![](//k.sinaimg.cn/n/sinakd20260418s/515/w1080h235/20260418/075b-e53d41570ee1b890e4c625b47fb483fa.png/w700d1q75cms.jpg)
NVIDA
SHEN ZHEN
![](//k.sinaimg.cn/n/sinakd20260418s/13/w1080h533/20260418/715d-a9108790bf6119247ca1d53846e012ae.jpg/w700d1q75cms.jpg)
引言/
Introduction
当RTX 4090还在刷新玩家对「旗舰显卡」的认知时，RTX 5090已携颠覆性升级强势登场！本期深度拆解两代神卡的硬核差异，帮你找到最适合自己的性能之选。
1
性能鸿沟：RTX 5090重新定义天花板
RTX 5090：未来五年的【全能怪兽】
算力暴增50%：搭载全新Blackwell架构，CUDA核心从16384个猛增至21760个，配合2.61GHz Boost频率（RTX 4090为2.23GHz），理论算力提升超50%！实测《赛博朋克2077》8K光追+DLSS 4.0模式下，RTX 5090稳跑120FPS，而RTX 4090仅65FPS。
显存全面进化：32GB GDDR7显存+1.1TB/s带宽（RTX 4090为24GB GDDR6X，带宽1TB/s），AI训练加载速度快25%，8K视频渲染效率提升3倍。
2
RTX 4090：4K游戏的【性价比之王】
01
经典架构依然能打
   16384个CUDA核心+24GB显存，在《荒野大镖客2》《永劫无间》等热门游戏中，4K分辨率下稳定80-100FPS，1440p更是轻松突破144FPS。
02
成熟技术无短板
   支持DLSS 3.5和第三代光追，《微软飞行模拟》4K光追下帧率比RTX 3090提升40%，且功耗控制更优（450W vs RTX 5090的575W）。
3
技术革命：RTX 5090的未来通行证
01
第五代光追+DLSS 4.0 
画质与帧率兼得
光追效率翻倍：RTX 5090的第五代RT Core实现光线追踪性能翻倍，水面反射、阴影渲染细节远超RTX 4090，搭配DLSS 4.0的「帧生成2.0」技术，4K 240FPS全光追成为现实。
AI加速质变：全新Tensor Core让Stable Diffusion出图速度提升3倍，8K影视剪辑实时预览无卡顿，创作者生产力直接起飞。
02
RTX 4090：经典技术的「够用哲学」
DLSS 3.5依然实用：在《黑神话：悟空》中开启DLSS 3.5后，4K帧率从45FPS提升至82FPS，画质损失几乎不可察觉。
功耗与散热更友好：450W功耗搭配三风扇六热管散热，普通ATX3.0电源即可稳定运行，装机成本更低。
4
选购指南：你的场景决定最优解
（1）选RTX 5090的三大理由
✅ 极致玩家：追求8K 120Hz光追、AI游戏模组开发；
✅ 专业创作者：3D建模、8K影视剪辑、AI训练刚需；
✅ 未来主义者：适配DLSS 4.0、多帧生成等下一代技术。
（2）选RTX 4090的三大理由
✅ 性价比之选：4K游戏性能90%接近RTX 5090，价格低30%；
✅ 实用主义者：日常游戏、直播推流、轻度渲染需求；
✅ 稳妥升级：成熟架构+稳定驱动，适合不想折腾的玩家。
3
四、英伟达深圳售后护航，安心无忧
终身质保：RTX 5090享核心硬件+散热模组全方位保障；
专属技术支持：AI驱动优化、超频问题1对1解决；
固件升级优先权：率先适配未来3年新游戏与创作软件。
总结：RTX 5090是「一步到位」的未来之选，RTX 4090是「务实之选」的性价比标杆。无论你选择哪款，英伟达深圳官方售后团队都将为你提供最专业的支持！
## [头条号入驻](//cj.sina.com.cn/index/guide)
## [电脑独立显卡和集成显卡哪个好 两者区别分析介绍](//t.cj.sina.com.cn/articles/view/7879848900/1d5acf3c401902xroe)
## [HUAWEI MatePad Pro 13.2 英寸](//t.cj.sina.com.cn/articles/view/7879848900/1d5acf3c401902xrnw)
## [如何快速阅读财务报表?](//cj.sina.com.cn/articles/view/7879848900/1d5acf3c401902xrnq)
## 财经自媒体联盟[更多自媒体作者](https://finance.sina.com.cn/cj/authorlist.shtml?fr=pc_cjarticle)
## 热文排行榜
![](//n.sinaimg.cn/finance/pc/cj/kandian/img/article_pic05.png)
[新浪财经头条意见反馈留言板](//news.sina.com.cn/feedback/post.html)
4001102288 欢迎批评指正
![](//beacon.sina.com.cn/a.gif?noScript)


"""
GENERATION_TEMPERATURE = 0.4


def _extract_score(logprobs_content: list[dict[str, Any]]) -> float:
    """Extract sigmoid(logit_yes - logit_no) from the first token's top_logprobs."""
    for entry in logprobs_content:
        yes_lp: float | None = None
        no_lp: float | None = None
        for cand in entry.get("top_logprobs") or []:
            tok = (cand.get("token") or "").strip().lower()
            if tok == "yes" and yes_lp is None:
                yes_lp = cand.get("logprob")
            elif tok == "no" and no_lp is None:
                no_lp = cand.get("logprob")
        if yes_lp is not None or no_lp is not None:
            lp_yes = yes_lp if yes_lp is not None else -100.0
            lp_no = no_lp if no_lp is not None else -100.0
            return 1.0 / (1.0 + math.exp(-(lp_yes - lp_no)))
    return 0.0


def score_and_generate(prompt: str, client: OpenAI) -> tuple[float, Iterator[str]]:
    """Return (relevance_score, text_stream).

    Call 1: greedy 1-token with logprobs → score + greedy yes/no decision.
    Call 2 (only when yes): streaming continuation with temperature sampling,
    reusing KV cache via cache_prompt so the second prefill is nearly free.
    """
    t0 = time.perf_counter()
    resp1 = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=1,
        logprobs=30,
        temperature=0.0,
        extra_body={"cache_prompt": True},
    )
    t1 = time.perf_counter()
    choice1 = resp1.choices[0]
    first_token: str = choice1.text
    print(f"[timing] call1 (prefill+1tok): {t1 - t0:.3f}s  token={first_token.strip()!r}")

    raw_lp = choice1.logprobs  # type: ignore[union-attr]
    content: list[dict[str, Any]] = []
    if raw_lp is not None:
        content = raw_lp.model_dump().get("content") or []
    score = _extract_score(content)

    if first_token.strip().lower() != "yes":
        return score, iter([first_token])

    t2 = time.perf_counter()
    stream = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt + first_token,
        max_tokens=8191,
        temperature=GENERATION_TEMPERATURE,
        stream=True,
        extra_body={"cache_prompt": True},
    )

    def _chunks() -> Iterator[str]:
        first = True
        for chunk in stream:
            text = chunk.choices[0].text
            if text:
                if first:
                    print(f"[timing] call2 TTFT:          {time.perf_counter() - t2:.3f}s")
                    first = False
                yield text
        print(f"[timing] call2 total:         {time.perf_counter() - t2:.3f}s")

    return score, itertools.chain([first_token], _chunks())


def main() -> None:
    client = OpenAI(base_url=SERVER_URL, api_key="not-needed")

    prompt = render_raw_prompt(
        query,
        document,
        instruction=TRAINING_INSTRUCTION,
        system_prompt=TRAINING_SYSTEM_PROMPT,
    )

    score, text_stream = score_and_generate(prompt, client)

    print(f"\nScore:    {score:.6f}")
    print("\nOutput:")
    for chunk in text_stream:
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
