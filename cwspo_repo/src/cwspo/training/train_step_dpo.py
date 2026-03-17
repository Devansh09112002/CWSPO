from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from cwspo.models.hf import load_causal_lm, resolve_device_map
from cwspo.schemas import PairRecord
from cwspo.training.dataset import PairDataset, collate_pairs
from cwspo.training.losses import set_pad_token_id, weighted_step_dpo_loss
from cwspo.utils.io import write_json


def build_policy_and_ref(cfg):
    policy, tokenizer = load_causal_lm(
        cfg.training.model_name,
        dtype=cfg.dtype,
        trust_remote_code=cfg.training.trust_remote_code,
        device_map=resolve_device_map(cfg.device.policy),
        load_in_4bit=cfg.training.load_in_4bit,
        attn_implementation=cfg.training.attn_implementation,
    )
    ref_model, _ = load_causal_lm(
        cfg.training.reference_model_name,
        dtype=cfg.dtype,
        trust_remote_code=cfg.training.trust_remote_code,
        device_map=resolve_device_map(cfg.device.policy),
        load_in_4bit=cfg.training.load_in_4bit,
        attn_implementation=cfg.training.attn_implementation,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.training.use_lora:
        if cfg.training.load_in_4bit:
            policy = prepare_model_for_kbit_training(policy)
        lora_cfg = LoraConfig(
            r=cfg.training.lora_r,
            lora_alpha=cfg.training.lora_alpha,
            lora_dropout=cfg.training.lora_dropout,
            target_modules=cfg.training.target_modules,
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_cfg)
        policy.print_trainable_parameters()

    return policy, ref_model, tokenizer


def train(cfg, rows: list[PairRecord]) -> dict:
    if not rows:
        summary = {
            "status": "skipped_no_pairs",
            "method_name": cfg.method.name,
            "num_pairs": 0,
            "num_steps": 0,
            "mean_loss": 0.0,
            "mean_dpo_term": 0.0,
            "mean_ref_pen": 0.0,
        }
        write_json(cfg.paths.train_metrics_file, summary)
        if cfg.paths.training_report_file:
            write_json(cfg.paths.training_report_file, summary)
        if cfg.paths.training_report_md_file:
            Path(cfg.paths.training_report_md_file).write_text(
                "# Training Report\n\nNo training pairs were available.\n",
                encoding="utf-8",
            )
        return summary

    start_time = time.time()
    policy, ref_model, tokenizer = build_policy_and_ref(cfg)
    set_pad_token_id(tokenizer.pad_token_id)

    ds = PairDataset(rows, tokenizer, max_length=cfg.training.max_length)
    dl = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer.pad_token_id),
    )

    total_steps = math.ceil(len(dl) / cfg.training.grad_accum_steps) * cfg.training.num_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)

    optim = AdamW(policy.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    policy.train()
    global_step = 0
    metrics: dict[str, list[float]] = {"loss": [], "dpo_term": [], "ref_pen": []}
    optimizer_step_logs: list[dict] = []

    for epoch in range(cfg.training.num_epochs):
        pbar = tqdm(dl, desc=f"Training epoch {epoch+1}/{cfg.training.num_epochs}")
        optim.zero_grad(set_to_none=True)
        steps_since_update = 0
        for step, batch in enumerate(pbar, start=1):
            loss, aux = weighted_step_dpo_loss(
                policy,
                ref_model,
                batch,
                beta=cfg.training.beta,
                lambda_ref=cfg.training.lambda_ref,
            )
            (loss / cfg.training.grad_accum_steps).backward()
            steps_since_update += 1

            metrics["loss"].append(float(loss.item()))
            metrics["dpo_term"].append(float(aux.get("dpo_term", 0.0)))
            if "ref_pen" in aux:
                metrics["ref_pen"].append(float(aux["ref_pen"]))

            if step % cfg.training.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                steps_since_update = 0
                if global_step == 1 or global_step % cfg.training.log_every == 0:
                    pbar.set_postfix(loss=float(loss.item()), step=global_step)
                optimizer_step_logs.append(
                    {
                        "optimizer_step": global_step,
                        "epoch": epoch + 1,
                        "loss": float(loss.item()),
                        "dpo_term": float(aux.get("dpo_term", 0.0)),
                        "ref_pen": float(aux.get("ref_pen", 0.0)),
                    }
                )

                if global_step % cfg.training.save_every == 0:
                    ckpt = Path(cfg.paths.checkpoint_dir) / f"step_{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

        if steps_since_update > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == 1 or global_step % cfg.training.log_every == 0:
                pbar.set_postfix(loss=float(loss.item()), step=global_step)
            optimizer_step_logs.append(
                {
                    "optimizer_step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "dpo_term": float(aux.get("dpo_term", 0.0)),
                    "ref_pen": float(aux.get("ref_pen", 0.0)),
                }
            )

    final_ckpt = Path(cfg.paths.checkpoint_dir) / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)

    wall_clock = time.time() - start_time
    summary = {
        "status": "completed",
        "method_name": cfg.method.name,
        "pair_type": rows[0].meta.get("pair_type") if rows else None,
        "num_pairs": len(rows),
        "effective_batch_size": int(cfg.training.batch_size * cfg.training.grad_accum_steps),
        "batch_size": cfg.training.batch_size,
        "grad_accum_steps": cfg.training.grad_accum_steps,
        "num_epochs": cfg.training.num_epochs,
        "num_batches": len(dl),
        "num_steps": global_step,
        "mean_loss": float(sum(metrics["loss"]) / max(1, len(metrics["loss"]))),
        "mean_dpo_term": float(sum(metrics["dpo_term"]) / max(1, len(metrics["dpo_term"]))),
        "mean_ref_pen": float(sum(metrics["ref_pen"]) / max(1, len(metrics["ref_pen"]))) if metrics["ref_pen"] else 0.0,
        "wall_clock_time_sec": float(wall_clock),
        "base_model_name": cfg.training.model_name,
        "reference_model_name": cfg.training.reference_model_name,
        "adapter_output_path": str(final_ckpt),
        "optimizer_step_logs": optimizer_step_logs,
    }
    write_json(cfg.paths.train_metrics_file, summary)
    if cfg.paths.training_report_file:
        write_json(cfg.paths.training_report_file, summary)
    if cfg.paths.training_report_md_file:
        lines = [
            "# Training Report",
            "",
            f"- Method: `{summary['method_name']}`",
            f"- Pair type: `{summary['pair_type']}`",
            f"- Num pairs: `{summary['num_pairs']}`",
            f"- Effective batch size: `{summary['effective_batch_size']}`",
            f"- Num optimizer steps: `{summary['num_steps']}`",
            f"- Mean loss: `{summary['mean_loss']}`",
            f"- Mean DPO term: `{summary['mean_dpo_term']}`",
            f"- Mean ref penalty: `{summary['mean_ref_pen']}`",
            f"- Wall clock seconds: `{summary['wall_clock_time_sec']}`",
            f"- Base model: `{summary['base_model_name']}`",
            f"- Adapter output: `{summary['adapter_output_path']}`",
            "",
        ]
        Path(cfg.paths.training_report_md_file).write_text("\n".join(lines), encoding="utf-8")
    return summary
