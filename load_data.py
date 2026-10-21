from pathlib import Path
import urllib.request

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

FILES = {
    "training_v2.csv": (
        "https://raw.githubusercontent.com/IshanaRaina/To-Live-or-Not-To-Live/"
        "master/Datasets/training_v2.csv"
    ),
    "unlabeled.csv": (
        "https://raw.githubusercontent.com/IshanaRaina/To-Live-or-Not-To-Live/"
        "master/Datasets/unlabeled.csv"
    ),
    "WiDS Datathon 2020 Dictionary.csv": (
        "https://raw.githubusercontent.com/IshanaRaina/To-Live-or-Not-To-Live/"
        "master/Datasets/WiDS%20Datathon%202020%20Dictionary.csv"
    ),
}


def ensure_data(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already present: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return data_dir


def load(data_dir: Path = DATA_DIR):
    ensure_data(data_dir)
    train = pd.read_csv(data_dir / "training_v2.csv")
    unlabeled = pd.read_csv(data_dir / "unlabeled.csv")
    dictionary = pd.read_csv(data_dir / "WiDS Datathon 2020 Dictionary.csv")
    return train, unlabeled, dictionary


if __name__ == "__main__":
    train, unlabeled, dictionary = load()
    print()
    print(f"train:      {train.shape[0]:,} rows x {train.shape[1]} columns")
    print(f"unlabeled:  {unlabeled.shape[0]:,} rows x {unlabeled.shape[1]} columns")
    print(f"dictionary: {dictionary.shape[0]:,} rows x {dictionary.shape[1]} columns")
    print()
    print("target hospital_death:")
    print(train["hospital_death"].value_counts().to_string())
    print()
    print("dictionary columns:", ", ".join(dictionary.columns.astype(str)))
