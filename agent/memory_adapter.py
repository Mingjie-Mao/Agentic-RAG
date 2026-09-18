"""Fail-closed adapter for Mingjie-Mao/llm-long-term-memory."""

import json
import uuid

import httpx

from app.config import settings


class MemoryUnavailable(RuntimeError):
    pass


class LongTermMemoryAdapter:
    def __init__(self, user, *, client=None):
        cfg = settings()
        self.url = cfg.memory_url.rstrip("/")
        self.namespace = f"{user.tenant_id}:{user.id}"
        try:
            tokens = json.loads(cfg.memory_tokens_json)
        except (TypeError, ValueError) as exc:
            raise MemoryUnavailable("长期记忆 token 配置无效") from exc
        self.token = tokens.get(self.namespace, "") if isinstance(tokens, dict) else ""
        self.client = client

    @property
    def enabled(self):
        return bool(self.url and self.token)

    def _post(self, path, body, headers=None):
        if not self.enabled:
            raise MemoryUnavailable("长期记忆未为当前用户配置受信命名空间")
        merged = {"Authorization": f"Bearer {self.token}"} | (headers or {})
        try:
            if self.client:
                response = self.client.post(path, json=body, headers=merged)
            else:
                response = httpx.post(
                    self.url + path,
                    json=body,
                    headers=merged,
                    timeout=settings().memory_timeout_seconds,
                    trust_env=False,
                )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MemoryUnavailable("长期记忆服务暂时不可用") from exc

    def search(self, query, limit=5):
        result = self._post(
            "/v1/memories/search",
            {
                "user_id": self.namespace,
                "query": query,
                "limit": min(max(limit, 1), 8),
                "explain": True,
            },
        )
        return {
            "memories": [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "status": row.get("status"),
                    "valid_from": row.get("valid_from"),
                    "valid_to": row.get("valid_to"),
                    "scope": row.get("scope"),
                    "source": row.get("source", {}),
                    "score": row.get("score"),
                }
                for row in result.get("memories", [])
            ],
            "candidates_considered": result.get("candidates_considered", 0),
        }

    def remember(self, content, session_id=None, idempotency_key=None):
        return self._post(
            "/v1/messages",
            {
                "user_id": self.namespace,
                "role": "user",
                "content": content,
                "session_id": session_id or f"agentic-rag-{uuid.uuid4()}",
            },
            {"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
