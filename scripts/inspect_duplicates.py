
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/movies_dataset")

files = {
    "metadata": ("movies_metadata.csv", "id"),
    "keywords": ("keywords.csv", "id"),
    "credits": ("credits.csv", "id")
}

for name, (filename, id_col) in files.items():

    df = pd.read_csv(
        DATA_PATH / filename,
        low_memory=False
    )

    df[id_col] = pd.to_numeric(
        df[id_col], errors="coerce"
    )

    duplicates = df[
        df[id_col].notna()
        & df[id_col].duplicated(keep=False)
    ]

    duplicate_groups = duplicates.groupby(id_col)

    identical = 0
    conflicting = 0

    for _, group in duplicate_groups:
        if len(group.drop_duplicates()) == 1:
            identical += 1
        else:
            conflicting += 1

    print(f"\n{name.upper()}")
    print("Identical duplicate groups:", identical)
    print("Conflicting duplicate groups:", conflicting)
