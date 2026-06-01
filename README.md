# langchain-pi

Native LangChain chat models for **OpenAI Codex** with ChatGPT Plus/Pro
subscription OAuth and **OpenCode Zen/Go**. No Pi runtime, no Node sidecar.

## Requirements

- Python >= 3.9
- For Codex: an `openai-codex` credential in `~/.pi/agent/auth.json`, or sign in
  with `codex-login`
- For paid OpenCode models: `OPENCODE_API_KEY` or an explicit `api_key`

## Install

```sh
pip install langchain-pi
# or: uv add langchain-pi
```

## Usage

`create_chat` routes the supported native providers:

```python
from langchain_pi import create_chat

codex = create_chat("openai-codex", "gpt-5.3-codex-spark")
free = create_chat("opencode", "deepseek-v4-flash-free")
go = create_chat("opencode-go", "glm-5", api_key="...")
```

Or construct Codex directly:

```python
from langchain_pi import ChatCodex

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
from langchain_pi import ChatCodex

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
from langchain_pi import ChatOpencode

free = ChatOpencode("deepseek-v4-flash-free")
paid = ChatOpencode("glm-5")
go = ChatOpencode("glm-5", tier="go")
```

Free models include `deepseek-v4-flash-free`, `big-pickle`, `mimo-v2.5-free`, and
`nemotron-3-super-free`.

## License

MIT
