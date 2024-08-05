"""Embedder."""

import os
from typing import Any

import pandas as pd
import tiktoken

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data/summary.parquet"))

embedding_model = "text-embedding-3-large"
encoding_name = "cl100k_base"
max_tokens = 8192

df["embedding_input"] = df["title"] + " " + df["content"]

columns = ["title", "url", "content", "summary", "embedding_input"]
new_df = pd.DataFrame(columns=columns)


def make_chunks(data: str | list[Any], length: int) -> list[str]:
    return [data[i : i + length] for i in range(0, len(data), length)]


encoding = tiktoken.get_encoding(encoding_name=encoding_name)

for idx, row in df.iterrows():
    embedding_input = row["embedding_input"]
    tokens = encoding.encode(text=embedding_input)

    for chunk in make_chunks(data=tokens, length=max_tokens):
        text = encoding.decode(tokens=chunk)


# with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
#     api_key = file.read().strip()

# client = OpenAIClient(api_key=api_key)

# result = client.create_embedding(text=text_8192, model="text-embedding-3-large")

# print(len(result))
