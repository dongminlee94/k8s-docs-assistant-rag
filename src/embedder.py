"""Embedder."""

import os

import pandas as pd
import tiktoken

from client import OpenAIClient

summary_df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data/summary.parquet"))

embedding_model = "text-embedding-3-large"
encoding_name = "cl100k_base"
max_tokens = 8192

summary_df["embedding_input"] = summary_df["title"] + " " + summary_df["content"]

columns = ["title", "url", "content", "summary", "embedding_input", "embedding_output"]
rows = []


def make_chunks(data: str | list[int], length: int) -> list[str]:
    return [data[i : i + length] for i in range(0, len(data), length)]


encoding = tiktoken.get_encoding(encoding_name=encoding_name)

for idx, row in summary_df.iterrows():
    embedding_input = row["embedding_input"]
    tokens = encoding.encode(text=embedding_input)

    for chunk in make_chunks(data=tokens, length=max_tokens):
        text = encoding.decode(tokens=chunk)

        rows.append(
            {
                "title": row["title"],
                "url": row["url"],
                "content": row["content"],
                "summary": row["summary"],
                "embedding_input": text,
                "embedding_output": None,
            },
        )

embedding_df = pd.DataFrame(data=rows, columns=columns)

embedding_df

print(embedding_df.shape)
print(embedding_df.columns)
print(embedding_df.head())

with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
    api_key = file.read().strip()

client = OpenAIClient(api_key=api_key)

for idx, row in embedding_df.iterrows():
    embedding_output = client.create_embedding(text=row["embedding_input"], model=embedding_model)
    embedding_df.loc[idx, "embedding_output"] = embedding_output

print(embedding_df.shape)
print(embedding_df.columns)
print(embedding_df.head())
