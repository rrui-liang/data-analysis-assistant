import json
from datasets import Dataset

def load_train_dataset(path="./data/train_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

if __name__ == "__main__":
    ds = load_train_dataset()
    print(ds[0])
