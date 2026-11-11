"""Shared helpers: where the server is and which model it serves."""

import os
import sys

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def base_url() -> str:
    return os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)


def pick_model(client) -> str:
    """Ask the server which model it serves (GET /v1/models) instead of hard-coding it."""
    try:
        return client.models.list().data[0].id
    except Exception as e:
        print(f"Could not reach {client.base_url} ({e.__class__.__name__}).\n"
              "Is the server running? Try: make deploy   (or ./fallback/ollama.sh up)")
        sys.exit(1)
