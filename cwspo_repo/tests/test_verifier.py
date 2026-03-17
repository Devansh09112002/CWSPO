import math

import torch

from cwspo.models.verifier import extract_process_reward_probability, split_prefix_lines


def test_split_prefix_lines_preserves_step_boundaries():
    prefix = "First compute 2 + 2 = 4.\n\nThen multiply by 3 to get 12.\n"
    assert split_prefix_lines(prefix) == [
        "First compute 2 + 2 = 4.",
        "Then multiply by 3 to get 12.",
    ]


def test_extract_process_reward_probability_uses_last_step_marker():
    logits = torch.tensor(
        [
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )
    token_mask = torch.tensor([[False, True, False, True]])

    reward = extract_process_reward_probability(
        logits,
        token_mask,
        positive_label_index=1,
    )

    expected = torch.softmax(torch.tensor([0.0, 3.0]), dim=0)[1].item()
    assert math.isclose(reward, expected, rel_tol=1e-6)
