from __future__ import annotations

from typing import Any, Dict

from social_itp.world_model.dummy_world_model import DummyWorldModel
from social_itp.world_model.prompt_builder import PromptBuilder, PromptBuildConfig
from social_itp.world_model.prompt_world_model import PromptWorldSimulator


def make_world_model(cfg: Dict[str, Any], llm_client=None):
    wm_type = str(cfg.get("type", "dummy")).lower()
    if wm_type == "dummy":
        return DummyWorldModel()
    if wm_type in {"prompt", "prompt_world_model"}:
        if llm_client is None:
            raise ValueError("Prompt world model requires an llm_client.")
        pb_cfg = cfg.get("prompt_builder", {}) or {}
        prompt_builder = PromptBuilder(
            PromptBuildConfig(
                max_graph_nodes=int(pb_cfg.get("max_graph_nodes", 30)),
                max_recent_utterances=int(pb_cfg.get("max_recent_utterances", 3)),
                max_text_len=int(pb_cfg.get("max_text_len", 220)),
                include_persona=bool(pb_cfg.get("include_persona", True)),
                include_bystanders=bool(pb_cfg.get("include_bystanders", True)),
                include_graph_edges=bool(pb_cfg.get("include_graph_edges", True)),
            )
        )
        return PromptWorldSimulator(llm_client=llm_client, prompt_builder=prompt_builder)
    raise ValueError(f"Unknown world model type: {wm_type}")
