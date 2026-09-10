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
free = create_chat("opencode", "nemotron-3.5-lightning-free")
go = create_chat("opencode-go", "minimax-m3", api_key="...")
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

Codex models: `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-6-astra`,
`gpt-5.5`, `gpt-5.3-codex-spark`. As of 2026-09 the ChatGPT account only serves
these; older GPT-5.x ids return `400 not supported when using Codex with a
ChatGPT account`.
Ids outside the catalog are still sent to the API, but `calculate_cost` prices
them as unknown and returns 0.

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

model = ChatCodex(model="gpt-5.5").bind_tools([get_weather])
msg = model.invoke("What's the weather in Paris?")
print(msg.tool_calls)
```

`tool_choice` is passed through to the Codex Responses API:

```python
forced = ChatCodex(model="gpt-5.5").bind_tools(
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

free = ChatOpencode("nemotron-3.5-lightning-free")
paid = ChatOpencode("minimax-m3", api_key="...")
go = ChatOpencode("minimax-m3", tier="go", api_key="...")
```

Free models include `nemotron-3.5-lightning-free`, `big-pickle`, and `mimo-v2.5-free`.

## Claude Code (Anthropic subscription)

`ChatClaudeCode` talks to the Anthropic Messages API authenticated with the
Claude Code OAuth session already on the machine (`~/.claude/.credentials.json`),
billing requests against your Claude Code subscription — no API key. It reads and
refreshes the token in place (with a `claude` CLI fallback).
On macOS it reads the login keychain item `Claude Code-credentials` first and
falls back to `~/.claude/.credentials.json`, writing the refreshed token back
to whichever source it read.
An explicit `creds_path` always wins and uses only that file.

```python
from open_langchain import ChatClaudeCode, create_chat

chat = create_chat("claude-code", "claude-sonnet-4-6")
print(chat.invoke("Hello!").content)

# Or construct directly, with options:
opus = ChatClaudeCode(model="claude-opus-5", reasoning="medium")
```

Models: `claude-opus-5`, `claude-fable-5-1`, `claude-fable-5`, `claude-opus-4-8`,
`claude-opus-4-7`, `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-haiku-4-5`.
Reasoning uses adaptive thinking on the Claude 5 family and Opus 4.8/4.7, and a
token budget on Sonnet 4.6 (Haiku has no reasoning). The Claude 5 models ship a
1M context window by default; for the older models the 1M-context beta is
**opt-in** via `long_context=True`, since the subscription rejects long-context
requests without extra credits otherwise. Tool calling and streaming work as with
`ChatCodex`.

`tool_choice` is forwarded to the Messages API unchanged. `claude-fable-5-1`
rejects a forced tool choice - `"any"` or a named tool - with a 400, so use
`tool_choice="auto"` with that model.

> Using a subscription OAuth session from a third-party app may violate
> Anthropic's terms and risk your account. See the
> [`pi-claude-code-auth`](https://github.com/cgaravitoq/pi-claude-code-auth)
> README before relying on this.

## License

MIT
