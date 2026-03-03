# Claude Code 指南

## 第三方文档查询

需要查询第三方库的 API 文档时，使用 context7 MCP 工具：

1. 先调用 `mcp__context7__resolve-library-id` 解析库名，获取 library ID
2. 再调用 `mcp__context7__query-docs` 查询具体文档

## 包管理

- 只用 `uv`，禁止用 `pip`
- 安装：`uv add package`
- 运行：`uv run tool`
- 禁止：`uv pip install`、`@latest` 语法

## 代码风格

- 命名：函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`
- 所有代码需要类型注解，公开 API 需要 docstring
- 行长最大 88 字符，用 f-string 做字符串格式化
- 函数保持小而专一，遵循现有代码模式

## 开发原则

- 优先简洁可读，避免过度设计
- 只修改与任务直接相关的代码
- 从最小功能开始，验证后再加复杂度
- 用早返回（early return）避免深层嵌套
- 保持核心逻辑干净，把实现细节推到边缘

## 测试(暂停使用)

本项目暂时不用pytest等测试工具，请使用example示例代码来代替


## 代码格式化工具

```bash
uv run ruff format .        # 格式化
uv run ruff check .         # 检查
uv run ruff check . --fix   # 自动修复
uv run pyright              # 类型检查
```

CI 修复顺序：格式化 → 类型错误 → Lint
