from __future__ import annotations

import random
import re
from dataclasses import dataclass

from datasets import load_dataset

from cwspo.schemas import ProcessGroundTruthRecord, PromptRecord


def parse_gsm8k_answer(text: str) -> tuple[str | None, str]:
    if "####" in text:
        solution, answer = text.rsplit("####", 1)
        return solution.strip() or None, answer.strip().replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return None, numbers[-1].replace(",", "") if numbers else text.strip()


def _format_prompt(question: str, append_step_by_step_suffix: bool) -> str:
    question = question.strip()
    if append_step_by_step_suffix and "step by step" not in question.lower():
        question = question + "\nShow your reasoning step by step."
    return question


def build_gsm8k_prompt_rows(
    *,
    split: str,
    count: int,
    seed: int,
    append_step_by_step_suffix: bool = True,
    dataset_name: str = "openai/gsm8k",
    dataset_config_name: str = "main",
) -> list[PromptRecord]:
    ds = load_dataset(dataset_name, dataset_config_name, split=split)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), min(count, len(ds))))
    rows: list[PromptRecord] = []
    for out_idx, ds_idx in enumerate(indices):
        row = ds[int(ds_idx)]
        solution, answer = parse_gsm8k_answer(row["answer"])
        rows.append(
            PromptRecord(
                id=f"{split.replace('[', '_').replace(']', '_').replace(':', '_')}_{out_idx:05d}",
                prompt=_format_prompt(row["question"], append_step_by_step_suffix),
                answer=answer,
                reference_solution=solution,
                source_split=split,
            )
        )
    return rows


@dataclass
class ProcessExample:
    prompt: str
    answer: str
    correct_steps: list[str]
    incorrect_steps: list[str]
    incorrect_final_answer: str
    gold_earliest_error_step: int


def _template_boxes(rng: random.Random) -> ProcessExample:
    boxes = rng.randint(3, 9)
    per_box = rng.randint(4, 12)
    given = rng.randint(2, min(10, boxes * per_box - 1))
    total = boxes * per_box
    left = total - given
    wrong_total = total + rng.choice([-per_box, per_box, 2])
    wrong_left = wrong_total - given
    return ProcessExample(
        prompt=f"A shelf has {boxes} boxes of markers with {per_box} markers in each box. Then {given} markers are given away. How many markers are left?",
        answer=str(left),
        correct_steps=[
            f"There are {boxes} boxes with {per_box} markers each, so {boxes} * {per_box} = {total} markers to start.",
            f"After giving away {given} markers, {total} - {given} = {left} markers remain.",
            f"The answer is {left}.",
        ],
        incorrect_steps=[
            f"There are {boxes} boxes with {per_box} markers each, so {boxes} * {per_box} = {wrong_total} markers to start.",
            f"After giving away {given} markers, {wrong_total} - {given} = {wrong_left} markers remain.",
            f"The answer is {wrong_left}.",
        ],
        incorrect_final_answer=str(wrong_left),
        gold_earliest_error_step=0,
    )


def _template_profit(rng: random.Random) -> ProcessExample:
    items = rng.randint(6, 20)
    sell = rng.randint(7, 18)
    cost = rng.randint(2, sell - 1)
    revenue = items * sell
    expense = items * cost
    profit = revenue - expense
    wrong_expense = expense + rng.choice([-items, items, 3])
    wrong_profit = revenue - wrong_expense
    return ProcessExample(
        prompt=f"A student sells {items} notebooks for ${sell} each after buying them for ${cost} each. What is the total profit?",
        answer=str(profit),
        correct_steps=[
            f"The revenue is {items} * {sell} = {revenue} dollars.",
            f"The total cost is {items} * {cost} = {expense} dollars.",
            f"The profit is {revenue} - {expense} = {profit} dollars.",
            f"The answer is {profit}.",
        ],
        incorrect_steps=[
            f"The revenue is {items} * {sell} = {revenue} dollars.",
            f"The total cost is {items} * {cost} = {wrong_expense} dollars.",
            f"The profit is {revenue} - {wrong_expense} = {wrong_profit} dollars.",
            f"The answer is {wrong_profit}.",
        ],
        incorrect_final_answer=str(wrong_profit),
        gold_earliest_error_step=1,
    )


def _template_distance(rng: random.Random) -> ProcessExample:
    speed1 = rng.randint(20, 55)
    hours1 = rng.randint(2, 5)
    speed2 = rng.randint(25, 60)
    hours2 = rng.randint(1, 4)
    first = speed1 * hours1
    second = speed2 * hours2
    total = first + second
    wrong_second = second + rng.choice([-speed2, speed2, 5])
    wrong_total = first + wrong_second
    return ProcessExample(
        prompt=f"A car travels {hours1} hours at {speed1} miles per hour and then {hours2} hours at {speed2} miles per hour. How many miles does it travel in total?",
        answer=str(total),
        correct_steps=[
            f"In the first part, the car travels {speed1} * {hours1} = {first} miles.",
            f"In the second part, the car travels {speed2} * {hours2} = {second} miles.",
            f"The total distance is {first} + {second} = {total} miles.",
            f"The answer is {total}.",
        ],
        incorrect_steps=[
            f"In the first part, the car travels {speed1} * {hours1} = {first} miles.",
            f"In the second part, the car travels {speed2} * {hours2} = {wrong_second} miles.",
            f"The total distance is {first} + {wrong_second} = {wrong_total} miles.",
            f"The answer is {wrong_total}.",
        ],
        incorrect_final_answer=str(wrong_total),
        gold_earliest_error_step=1,
    )


def _template_bags(rng: random.Random) -> ProcessExample:
    apples1 = rng.randint(12, 30)
    apples2 = rng.randint(10, 28)
    bag_size = rng.choice([2, 4, 5, 6, 8])
    total = apples1 + apples2
    bags = total // bag_size
    wrong_bags = bags + rng.choice([-1, 1, 2])
    return ProcessExample(
        prompt=f"A farmer picks {apples1} apples in the morning and {apples2} apples in the afternoon. He packs them equally into bags of {bag_size}. How many full bags can he make?",
        answer=str(bags),
        correct_steps=[
            f"The farmer picked {apples1} + {apples2} = {total} apples altogether.",
            f"Each bag holds {bag_size} apples, so {total} // {bag_size} = {bags} full bags.",
            f"The answer is {bags}.",
        ],
        incorrect_steps=[
            f"The farmer picked {apples1} + {apples2} = {total} apples altogether.",
            f"Each bag holds {bag_size} apples, so {total} // {bag_size} = {wrong_bags} full bags.",
            f"The answer is {wrong_bags}.",
        ],
        incorrect_final_answer=str(wrong_bags),
        gold_earliest_error_step=1,
    )


def _template_perimeter(rng: random.Random) -> ProcessExample:
    length = rng.randint(5, 18)
    width = rng.randint(3, 14)
    twice_length = 2 * length
    twice_width = 2 * width
    perimeter = twice_length + twice_width
    wrong_perimeter = perimeter + rng.choice([-2, 2, 4])
    return ProcessExample(
        prompt=f"A rectangle has length {length} and width {width}. What is its perimeter?",
        answer=str(perimeter),
        correct_steps=[
            f"Twice the length is 2 * {length} = {twice_length}.",
            f"Twice the width is 2 * {width} = {twice_width}.",
            f"The perimeter is {twice_length} + {twice_width} = {perimeter}.",
            f"The answer is {perimeter}.",
        ],
        incorrect_steps=[
            f"Twice the length is 2 * {length} = {twice_length}.",
            f"Twice the width is 2 * {width} = {twice_width}.",
            f"The perimeter is {twice_length} + {twice_width} = {wrong_perimeter}.",
            f"The answer is {wrong_perimeter}.",
        ],
        incorrect_final_answer=str(wrong_perimeter),
        gold_earliest_error_step=2,
    )


PROCESS_TEMPLATES = [
    _template_boxes,
    _template_profit,
    _template_distance,
    _template_bags,
    _template_perimeter,
]


def build_process_eval_rows(*, count: int, seed: int) -> list[ProcessGroundTruthRecord]:
    rng = random.Random(seed)
    rows: list[ProcessGroundTruthRecord] = []
    for idx in range(count):
        example = rng.choice(PROCESS_TEMPLATES)(rng)
        rows.append(
            ProcessGroundTruthRecord(
                id=f"process_{idx:05d}",
                gold_earliest_error_step=example.gold_earliest_error_step,
                prompt=example.prompt,
                answer=example.answer,
                correct_steps=example.correct_steps,
                incorrect_steps=example.incorrect_steps,
                incorrect_final_answer=example.incorrect_final_answer,
            )
        )
    return rows
