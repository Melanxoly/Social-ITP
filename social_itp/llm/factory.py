from __future__ import annotations

from typing import Any, Dict

from social_itp.llm.client import DummyLLMClient, QwenClientConfig, QwenJSONClient


def make_llm_client(cfg: Dict[str, Any]):
    backend = str(cfg.get("backend", "dummy")).lower()
    if backend in {"dummy", "none", "local"}:
        return DummyLLMClient()
    if backend in {"qwen", "dashscope"}:
        return QwenJSONClient(
            QwenClientConfig(
                model_name=cfg.get("model_name", "qwen-plus-latest"),
                max_retries=int(cfg.get("max_retries", 2)),
                retry_sleep_sec=float(cfg.get("retry_sleep_sec", 1.5)),
                request_timeout_sec=float(cfg.get("request_timeout_sec", 120.0)),
                seed=cfg.get("seed", 42),
                api_key_env=cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
                base_url_env=cfg.get("base_url_env", "DASHSCOPE_BASE_URL"),
            )
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
