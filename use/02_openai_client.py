"""Stage 3b: the official OpenAI Python client, pointed at your own server.

The only change from calling OpenAI's API is `base_url`. Everything else
(chat, streaming, tool calling) is the same code.

    uv run python use/02_openai_client.py
    LLM_BASE_URL=http://localhost:11434/v1 uv run python use/02_openai_client.py   # Ollama
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from common import base_url, pick_model  # noqa: E402

client = OpenAI(base_url=base_url(), api_key="not-needed-locally")  # ← the one-line change


def answer(r) -> str:
    """The reply text. Thinking models can use up max_tokens before answering."""
    text = r.choices[0].message.content
    return text.strip() if text else "(no answer: the model used all max_tokens thinking; raise max_tokens)"



model = pick_model(client)
print(f"Server: {base_url()}\nModel:  {model}\n")

# --- 1. A plain chat completion ---------------------------------------------------
print("== 1. Chat completion ==")
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Name three things vLLM does to serve LLMs fast. One line each."}],
    max_tokens=2000,
)
print(answer(r))
print(f"({r.usage.completion_tokens} tokens)\n")

# --- 2. Streaming: tokens arrive as they are generated -----------------------------
print("== 2. Streaming ==")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from 1 to 10, separated by spaces. Only the numbers."}],
    max_tokens=800,
    stream=True,
)
started = False
for chunk in stream:
    text = chunk.choices[0].delta.content if chunk.choices else None
    if text:
        if not started:  # skip leading blank lines left over from the model's thinking phase
            text, started = text.lstrip(), bool(text.strip())
        print(text, end="", flush=True)
print("\n")

# --- 3. Tool calling: the model asks *us* to run a function -----------------------
print("== 3. Tool calling ==")


def get_weather(city: str) -> dict:
    """A stand-in for a real API call."""
    return {"city": city, "temperature_c": 4, "conditions": "windy"}


tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What's the weather in Chicago right now?"}]

r = client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=1000)
msg = r.choices[0].message
if not msg.tool_calls:
    print("The model answered without calling the tool. Was the server started with\n"
          "--enable-auto-tool-choice and the right --tool-call-parser?")
    print(answer(r))
    sys.exit(1)

call = msg.tool_calls[0]
args = json.loads(call.function.arguments)
print(f"Model asked to call: {call.function.name}({args})")
result = get_weather(**args)
print(f"We ran it and got:   {result}")

messages += [msg, {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}]
r = client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=1000)
print(f"Final answer:        {answer(r)}")
