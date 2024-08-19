import os
from collections import Counter

import pandas as pd

file_path = os.path.join(os.path.dirname(__file__), "..", "data/vector_db.parquet")
df = pd.read_parquet(file_path)

print(df.shape)

subdirs = [row.url.split("/")[4] for row in df.itertuples()]
count = Counter(subdirs)

print(count)
