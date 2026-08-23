# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat client: urllib, non-streaming, retries, wall-clock.

Engine-agnostic by design: boyle, mlx-lm server, or a cloud
endpoint — anything speaking /v1/chat/completions. Cost is metered here:
wall-clock around the call plus the server's token usage. Non-streaming
keeps accounting exact; the measurement harness runs batches, not chats.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    finish_reason: str
    token_entropies: list | None = None

    @property
    def mean_entropy(self) -> float | None:
        te = self.token_entropies
        return (sum(te) / len(te)) if te else None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class ChatClient:
    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None, timeout_s: int = 900):
        if base_url is None:
            base_url = os.environ.get("CLAUSIUS_ENDPOINT",
                                      "http://127.0.0.1:11434/v1")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(self, messages: list[dict], max_tokens: int,
                 temperature: float = 0.0, enable_thinking: bool = False,
                 signal: bool = False, seed: int | None = None,
                 retries: int = 3,
                 template_kwargs: dict | None = None) -> Completion:
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if enable_thinking:
            body["enable_thinking"] = True  # boyle passthrough; ignored elsewhere
        if template_kwargs:
            # vLLM / llama.cpp / boyle convention: per-model template knobs
            # (Qwen3.8 reasoning_effort, ...) ride chat_template_kwargs
            body["chat_template_kwargs"] = dict(template_kwargs)
        if signal:
            # boyle extension: token_entropies ride the logprobs field
            body["logprobs"] = True
            body["top_logprobs"] = 0
        if seed is not None:
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body).encode(), headers=headers)
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    out = json.loads(r.read())
                wall = time.perf_counter() - t0
                choice = out["choices"][0]
                usage = out.get("usage") or {}
                ents = (choice.get("logprobs") or {}).get("token_entropies")
                return Completion(
                    token_entropies=ents,
                    text=choice["message"].get("content") or "",
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    wall_s=wall,
                    finish_reason=choice.get("finish_reason") or "stop",
                )
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"chat endpoint failed after {retries} tries: {last}")
