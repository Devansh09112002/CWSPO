from cwspo.pipeline.build_pairs import first_divergence
from cwspo.utils.steps import split_steps


def test_split_steps_handles_numbered_lines():
    reasoning = "Step 1: Add 3 and 5.\nStep 2: The total is 8."
    assert split_steps(reasoning) == ["Add 3 and 5.", "The total is 8."]


def test_split_steps_handles_inline_markers():
    reasoning = "Step 1: Multiply 4 by 6. Step 2: The answer is 24."
    assert split_steps(reasoning) == ["Multiply 4 by 6.", "The answer is 24."]


def test_first_divergence_detects_content_change():
    a = ["Compute 3+5.", "Result is 8."]
    b = ["Compute 3+5.", "Result is 7."]
    assert first_divergence(a, b) == 1


def test_first_divergence_detects_prefix_length_change():
    a = ["Compute 3+5."]
    b = ["Compute 3+5.", "Result is 8."]
    assert first_divergence(a, b) == 1
