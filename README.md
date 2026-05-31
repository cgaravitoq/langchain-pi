# langchain-pi

A LangChain [`BaseChatModel`](https://python.langchain.com/) adapter for **Pi**
([`@earendil-works/pi-ai`](https://www.npmjs.com/package/@earendil-works/pi-ai)),
the Python twin of [`langchain-pi-ts`](https://github.com/cgaravitoq/langchain-pi-ts).
Use Pi — and any provider, model and credential it resolves — from LangChain and
LangGraph in Python, with native tool calling and streaming.

Pi is a TypeScript/Node stack with no Python SDK, so `langchain-pi` drives a tiny
**Node sidecar** that wraps pi-ai's `streamSimple` + `ModelRegistry`/`AuthStorage`.
Node keeps owning provider dispatch, OAuth refresh and the provider stealth
headers; Python only frames the request and reconstructs the streamed events.
**Python never touches a token.**

## Requirements

- **Python** ≥ 3.9
- **Node** ≥ 22.19.0 on `PATH`, with `@earendil-works/pi-ai` and
  `@earendil-works/pi-coding-agent` resolvable (install them where the sidecar can
  reach them, or point `node_modules_dir` at a `node_modules` directory that has
  them). Note: `NODE_PATH` does not work for ESM, so `node_modules_dir` resolves
  each package via its `package.json` entry.
- A provider authenticated in `~/.pi` (e.g. `openai-codex`), exactly as for the
  Pi CLI / the TS package.

## Install

```sh
pip install langchain-pi          # or: uv add langchain-pi
npm install @earendil-works/pi-ai @earendil-works/pi-coding-agent
```

## Usage

```python
from langchain_pi import ChatPi

model = ChatPi(
    provider="openai-codex",
    model="gpt-5.3-codex-spark",
    reasoning="minimal",
    system="You are a helpful assistant.",
)

print(model.invoke("Hello!").content)
```

### Tool calling

Accepts any LangChain tool; schemas are converted to the JSON Schema pi-ai expects.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}, 24C."

agent_model = model.bind_tools([get_weather])
```

### Streaming

```python
for chunk in model.stream("Write a haiku."):
    print(chunk.content, end="")
```

### LangGraph

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, [get_weather])
agent.invoke({"messages": [("user", "What's the weather in Paris?")]})
```

### Pointing at your Node install

If pi-ai isn't resolvable from the sidecar's location, pass the `node_modules`
directory that contains it:

```python
ChatPi(provider="opencode", model="deepseek-v4-flash-free",
       node_modules_dir="/path/to/your/project/node_modules")
```

## Notes

- Node is a runtime prerequisite — the heavy provider/auth logic lives in pi-ai.
- Tool-call deltas and usage/cost metadata are reconstructed 1:1 with the TS twin.
- Cancellation (e.g. LangGraph) aborts the in-flight provider request via the sidecar.

## License

MIT
