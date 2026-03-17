from __future__ import annotations

import torch
import torch.nn.functional as F


_PAD_TOKEN_ID = None


def set_pad_token_id(pad_token_id: int) -> None:
    global _PAD_TOKEN_ID
    _PAD_TOKEN_ID = pad_token_id


def seq_logprob(model, prefix_ids: torch.Tensor, segment_ids: torch.Tensor):
    if _PAD_TOKEN_ID is None:
        raise RuntimeError("Pad token id not set; call set_pad_token_id first.")

    bsz = prefix_ids.shape[0]
    logp_sums = []
    tokenwise = []
    device = next(model.parameters()).device

    for i in range(bsz):
        p = prefix_ids[i][prefix_ids[i] != _PAD_TOKEN_ID].to(device)
        s = segment_ids[i][segment_ids[i] != _PAD_TOKEN_ID].to(device)
        full = torch.cat([p, s], dim=0).unsqueeze(0)
        if full.shape[1] < 2:
            logp_sums.append(torch.tensor(0.0, device=device))
            tokenwise.append(torch.zeros((1,), device=device))
            continue
        out = model(full)
        logits = out.logits[:, :-1, :]
        target = full[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        seg_tok_lp = tok_lp[:, p.shape[0] - 1 :] if p.shape[0] >= 1 else tok_lp
        logp_sums.append(seg_tok_lp.sum())
        tokenwise.append(seg_tok_lp.squeeze(0))

    return torch.stack(logp_sums), tokenwise


def weighted_step_dpo_loss(policy, ref_model, batch, beta: float = 0.1, lambda_ref: float = 0.0):
    lp_pref, tok_lp_pref = seq_logprob(policy, batch["prefix_ids"], batch["pref_ids"])
    lp_disp, tok_lp_disp = seq_logprob(policy, batch["prefix_ids"], batch["disp_ids"])

    with torch.no_grad():
        ref_lp_pref, ref_tok_lp_pref = seq_logprob(ref_model, batch["prefix_ids"], batch["pref_ids"])
        ref_lp_disp, ref_tok_lp_disp = seq_logprob(ref_model, batch["prefix_ids"], batch["disp_ids"])

    delta = (lp_pref - lp_disp) - (ref_lp_pref - ref_lp_disp)
    dpo_term = -F.logsigmoid(beta * delta)
    w = batch["weights"].to(dpo_term.device)

    if lambda_ref <= 0:
        return (w * dpo_term).mean(), {"dpo_term": float(dpo_term.mean().item())}

    penalties = []
    for a, b, ra, rb in zip(tok_lp_pref, tok_lp_disp, ref_tok_lp_pref, ref_tok_lp_disp):
        pa = ((a - ra) ** 2).mean()
        pb = ((b - rb) ** 2).mean()
        penalties.append(0.5 * (pa + pb))
    ref_pen = torch.stack(penalties)
    loss = (w * dpo_term + lambda_ref * (1.0 - w) * ref_pen).mean()
    return loss, {
        "dpo_term": float(dpo_term.mean().item()),
        "ref_pen": float(ref_pen.mean().item()),
    }
