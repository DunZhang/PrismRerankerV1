import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock

# ====== 配置区 ======
MODE = "new"  # "new" = 新建会话, "resume" = 恢复会话
RESUME_SESSION_ID = "7870c941-62e8-434e-a9af-4d1f7acc149b"  # 恢复时填入之前打印的 session_id
PROMPTS = ["介绍下爱因斯坦", "你简单说下他最最最重要的一个贡献吧，只说一个"]
# PROMPTS = ["我一共问了哪几个问题"]
# ====================

async def run():
    opts = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        resume=RESUME_SESSION_ID if MODE == "resume" else None,
        continue_conversation=MODE == "resume",
    )
    async with ClaudeSDKClient(options=opts) as client:
        for p in PROMPTS:
            print(f"\n>>> {p}")
            await client.query(p)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            print(b.text)
                if isinstance(msg, ResultMessage):
                    print(f"\n[session_id: {msg.session_id}]")

asyncio.run(run())