#!/usr/bin/env bash
# Stage 3a: talk to the model with nothing but HTTP.
# vLLM (and Ollama, and llama.cpp) speak the OpenAI API, so this is the whole protocol.
set -euo pipefail
BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"

echo "== GET $BASE_URL/models  (which model is being served?) =="
if ! MODELS=$(curl -sf "$BASE_URL/models"); then
  echo "Could not reach $BASE_URL. Is the server running? Try: make deploy"
  exit 1
fi
echo "$MODELS" | python3 -c '
import sys, json
for m in json.load(sys.stdin)["data"]:
    print(json.dumps({k: m.get(k) for k in ("id", "object", "owned_by", "max_model_len")}, indent=2))'
MODEL=$(echo "$MODELS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"][0]["id"])')

BODY=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "user", "content": "In two sentences: why does a GPU need memory for a KV cache?"}
  ],
  "max_tokens": 1000
}
EOF
)

echo
echo "== POST $BASE_URL/chat/completions =="
echo "$BODY"
echo
echo "== Response =="
curl -sf "$BASE_URL/chat/completions" -H "Content-Type: application/json" -d "$BODY" \
  | python3 -c '
import sys, json
r = json.load(sys.stdin)
msg = r["choices"][0]["message"]
if msg.get("reasoning_content") or msg.get("reasoning"):
    print("(the model also returned its reasoning in a separate field: reasoning_content)\n")
if not msg.get("content"):
    print("(no answer: the model spent all max_tokens thinking. Raise max_tokens and try again.)")
else:
    print(msg["content"].strip())
u = r["usage"]
print("\ntokens: %d in, %d out" % (u["prompt_tokens"], u["completion_tokens"]))
'
