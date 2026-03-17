from types import SimpleNamespace

import torch

from cwspo.training.losses import set_pad_token_id, weighted_step_dpo_loss


class ToyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=self.proj(hidden))


def test_weighted_step_dpo_loss_forward_backward():
    policy = ToyLM()
    ref_model = ToyLM()
    set_pad_token_id(0)

    batch = {
        "prefix_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "pref_ids": torch.tensor([[6, 7], [6, 0]], dtype=torch.long),
        "disp_ids": torch.tensor([[8, 9], [7, 0]], dtype=torch.long),
        "weights": torch.tensor([1.0, 0.4], dtype=torch.float32),
    }

    loss, aux = weighted_step_dpo_loss(policy, ref_model, batch, beta=0.2, lambda_ref=0.1)
    assert torch.isfinite(loss)
    assert "dpo_term" in aux
    assert "ref_pen" in aux

    loss.backward()
    grad_norm = 0.0
    for param in policy.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.abs().sum().item())
    assert grad_norm > 0.0
