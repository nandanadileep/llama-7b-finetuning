import yaml
from datasets import load_dataset

def load_dataset_from_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset(config["source"])
    return dataset["train"], config 

