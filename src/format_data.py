import json
from load_data import load_dataset_from_config


def normalize_example(example, field_map):
    try:
        return {
            "instruction": example[field_map["instruction"]],
            "input": example.get(field_map.get("input", ""), ""),
            "output": example[field_map["output"]],
        }
    except KeyError as e:
        raise ValueError(f"Dataset missing required field: {e}")


def build_prompt(example, style="alpaca"):
    if style == "alpaca":
        if example["input"].strip():
            return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        else:
            return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

    raise ValueError(f"Unknown prompt style: {style}")


def preprocess(config_path, output_path):
    dataset, cfg = load_dataset_from_config(config_path)

    field_map = cfg["fields"]
    style = cfg["prompt_style"]

    with open(output_path, "w") as f:
        for ex in dataset:
            norm = normalize_example(ex, field_map)
            text = build_prompt(norm, style)
            f.write(json.dumps({"text": text}) + "\n")


if __name__ == "__main__":
    preprocess(
        config_path="configs/dataset.yaml",
        output_path="data/processed/train.jsonl"
    )
