from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from social_itp.data.loader import DatasetLoadConfig, iter_loaded_examples
from social_itp.evaluator.factory import make_evaluator
from social_itp.experiment.config import save_json
from social_itp.experiment.recorder import JsonlRecorder
from social_itp.llm.factory import make_llm_client
from social_itp.policy.factory import make_policy
from social_itp.world_model.factory import make_world_model


def _as_policy_specs(raw: Any) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, str):
            specs.append({"name": item})
        elif isinstance(item, dict):
            specs.append(item)
        else:
            raise TypeError(f"Invalid policy spec: {item!r}")
    return specs


def _make_dataset_cfg(cfg: Dict[str, Any]) -> DatasetLoadConfig:
    d = cfg.get("data", {}) or {}
    return DatasetLoadConfig(
        data_root=d.get("data_root", "../data/MCSD"),
        topics=list(d.get("topics", [])),
        max_examples_per_topic=d.get("max_examples_per_topic"),
        max_context_nodes=int(d.get("max_context_nodes", 60)),
        max_reply_depth=int(d.get("max_reply_depth", 1)),
        reply_window_hours=int(d.get("reply_window_hours", 48)),
        min_replies=int(d.get("min_replies", 1)),
        num_bystanders=int(d.get("num_bystanders", 3)),
        max_node_text=int(d.get("max_node_text", 240)),
        min_action_text_len=int(d.get("min_action_text_len", 2)),
        filter_noise_comments=bool(d.get("filter_noise_comments", True)),
        filter_active_subreddits=bool(d.get("filter_active_subreddits", True)),
        max_active_subreddits=int(d.get("max_active_subreddits", 8)),
        bigfive_include_reason=bool(d.get("bigfive_include_reason", False)),
    )


def run_experiment(cfg: Dict[str, Any]) -> Path:
    exp_id = cfg.get("exp_id") or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    seed = int(cfg.get("seed", 42))
    output_dir = Path(cfg.get("output_dir", f"outputs/{exp_id}"))
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    logs_dir = output_dir / "logs"
    cases_dir = output_dir / "cases"
    for p in [raw_dir, summary_dir, logs_dir, cases_dir]:
        p.mkdir(parents=True, exist_ok=True)

    manifest = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "config": cfg,
    }
    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "config_resolved.json", cfg)

    llm_cfg = dict(cfg.get("llm", {}) or {})
    llm_cfg.setdefault("seed", seed)
    llm_client = make_llm_client(llm_cfg)
    world_model = make_world_model(cfg.get("world_model", {}) or {}, llm_client=llm_client)
    evaluator = make_evaluator(cfg.get("evaluator", {}) or {})

    policy_specs = _as_policy_specs(cfg.get("policies", ["random", "reactive"]))
    policies = [
        make_policy(
            spec["name"],
            spec,
            llm_client=llm_client,
            world_model=world_model,
            evaluator=evaluator,
            seed=seed,
        )
        for spec in policy_specs
    ]

    dataset_cfg = _make_dataset_cfg(cfg)
    if not dataset_cfg.topics:
        raise ValueError("No topics specified in config.data.topics")

    result_path = raw_dir / "results.jsonl"
    n_records = 0
    n_examples = 0

    with JsonlRecorder(result_path) as rec:
        for loaded in iter_loaded_examples(dataset_cfg):
            ex = loaded.example
            n_examples += 1
            for policy in policies:
                t0 = time.time()
                decision = policy.choose(ex)
                t_decision = time.time() - t0

                t1 = time.time()
                prediction = world_model.predict(ex.observation, decision.action)
                t_wm = time.time() - t1
                wm_raw = getattr(world_model, "last_raw_response", None)

                t2 = time.time()
                score = evaluator.score(ex.observation, decision.action, prediction)
                t_eval = time.time() - t2

                record = {
                    "exp_id": exp_id,
                    "seed": seed,
                    "topic": loaded.topic,
                    "thread_path": str(loaded.thread_path),
                    "example_index": loaded.example_index,
                    "example_id": ex.example_id,
                    "policy": getattr(policy, "name", policy.__class__.__name__),
                    "observation": ex.observation.to_dict(),
                    "gold_action": ex.action.to_dict(),
                    "gold_next_replies": [r.to_dict() for r in ex.label.next_replies],
                    "decision": decision.to_dict(),
                    "world_model_prediction": prediction.to_dict(),
                    "world_model_raw_response": wm_raw,
                    "evaluation": score.to_dict(),
                    "timing": {
                        "decision_sec": t_decision,
                        "world_model_sec": t_wm,
                        "evaluator_sec": t_eval,
                        "total_sec": t_decision + t_wm + t_eval,
                    },
                }
                rec.write(record)
                n_records += 1
                print(f"[OK] {n_records}: topic={loaded.topic} policy={record['policy']} example={ex.example_id}")

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["num_examples"] = n_examples
    manifest["num_records"] = n_records
    manifest["result_path"] = str(result_path)
    save_json(output_dir / "manifest.json", manifest)
    print(f"\n[OK] experiment finished: {output_dir}")
    print(f"[OK] records: {result_path}")
    return output_dir
