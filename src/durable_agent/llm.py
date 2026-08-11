from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from durable_agent.config import Settings, load_dotenv


class OpenAICompatibleClient:
    def __init__(self, settings: Settings | None = None) -> None:
        load_dotenv()
        self.settings = settings or Settings.load()
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")

    def complete_json(self, system: str, user: str, timeout: float = 60) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured. Use --mode deterministic or create .env.")
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model HTTP {exc.code}: {detail[-1000:]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Model network error: {exc.reason}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Model returned an invalid structured response.") from exc
