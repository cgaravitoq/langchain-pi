# open-langchain

Native LangChain chat models for **OpenAI Codex** (ChatGPT Plus/Pro
subscription OAuth), **Claude** (Claude Code subscription OAuth), and
**OpenCode Zen/Go**. No Pi runtime, no Node sidecar.

## Requirements

- Python >= 3.9
- For Codex: an `openai-codex` credential in `~/.pi/agent/auth.json`, or sign in
  with `codex-login`
- For Claude Code: the Claude Code CLI logged in once (`~/.claude/.credentials.json`)
- For paid OpenCode models: `OPENCODE_API_KEY` or an explicit `api_key`

## Install

```sh
pip install open-langchain
# or: uv add open-langchain
```

## Usage

`create_chat` routes the supported native providers:

```python
from open_langchain import create_chat

codex = create_chat("openai-codex", "gpt-5.3-codex-spark")
claude = create_chat("claude-code", "claude-sonnet-4-6")
free = create_chat("opencode", "deepseek-v4-flash-free")
go = create_chat("opencode-go", "glm-5", api_key="...")
```

Or construct Codex directly:

```python
from open_langchain import ChatCodex

model = ChatCodex(
    model="gpt-5.3-codex-spark",
    reasoning="minimal",
    system="You are a helpful assistant.",
)

print(model.invoke("Hello!").content)
```

## Codex Auth

`ChatCodex` reads the same `~/.pi/agent/auth.json` credential shape under
`openai-codex`, refreshes the OAuth token in place, and talks directly to
`https://chatgpt.com/backend-api/codex/responses`.

If no credential exists:

```sh
codex-login
codex-login --device
```

## Tool Calling

```python
from langchain_core.tools import tool
from open_langchain import ChatCodex

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}, 24C."

model = ChatCodex(model="gpt-5.3-codex").bind_tools([get_weather])
msg = model.invoke("What's the weather in Paris?")
print(msg.tool_calls)
```

`tool_choice` is passed through to the Codex Responses API:

```python
forced = ChatCodex(model="gpt-5.3-codex").bind_tools(
    [get_weather],
    tool_choice={"type": "function", "name": "get_weather"},
)
```

For agent loops, keep `tool_choice="auto"` unless every model turn should call the
same tool.

## Streaming

```python
for chunk in model.stream("Write a haiku."):
    print(chunk.content, end="")
```

## OpenCode

`ChatOpencode` uses OpenCode's OpenAI-compatible endpoints through
`langchain-openai`.

```python
from open_langchain import ChatOpencode

free = ChatOpencode("deepseek-v4-flash-free")
paid = ChatOpencode("glm-5")
go = ChatOpencode("glm-5", tier="go")
```

Free models include `deepseek-v4-flash-free`, `big-pickle`, `mimo-v2.5-free`, and
`nemotron-3-super-free`.

## Claude Code (Anthropic subscription)

`ChatClaudeCode` talks to the Anthropic Messages API authenticated with the
Claude Code OAuth session already on the machine (`~/.claude/.credentials.json`),
billing requests against your Claude Code subscription — no API key. It reads and
refreshes the token in place (with a `claude` CLI fallback).

```python
from open_langchain import ChatClaudeCode, create_chat

chat = create_chat("claude-code", "claude-sonnet-4-6")
print(chat.invoke("Hello!").content)

# Or construct directly, with options:
opus = ChatClaudeCode(model="claude-opus-5", reasoning="medium")
```

Models: `claude-opus-5`, `claude-fable-5`, `claude-opus-4-8`, `claude-opus-4-7`,
`claude-sonnet-5`, `claude-sonnet-4-6`, `claude-haiku-4-5`. Reasoning uses
adaptive thinking on the Claude 5 family and Opus 4.8/4.7, and a token budget on
Sonnet 4.6 (Haiku has no reasoning). The Claude 5 models ship a 1M context
window by default; for the older models the 1M-context beta is **opt-in** via
`long_context=True`, since the subscription rejects long-context requests
without extra credits otherwise. Tool calling and streaming work as with
`ChatCodex`.

> Using a subscription OAuth session from a third-party app may violate
> Anthropic's terms and risk your account. See the
> [`pi-claude-code-auth`](https://github.com/cgaravitoq/pi-claude-code-auth)
> README before relying on this.

## License

MIT
