"""Embedder."""

import os

import pandas as pd

# import tiktoken

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data/summary.parquet"))

print(df.shape)

# embedding_model = "text-embedding-3-large"
# max_tokens = 8192

# df["embedding_input"] = df["title"] + " " + df["content"]

# columns = ["title", "url", "content", "summary", "embedding_input"]
# new_df = pd.DataFrame(columns=columns)


# def make_chunks(content: str, length: int) -> list[str]:
#     return [content[i : i + length] for i in range(0, len(content), length)]


# tokenizer = tiktoken.get_encoding("cl100k_base")

# for idx, row in df.iterrows():
#     if len(row["embedding_input"]) >= max_tokens:
#         print(row["title"])
#         print(len(row["embedding_input"]))
#         print(len(tokenizer.encode(row["embedding_input"])))
#         print()


# with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
#     api_key = file.read().strip()

# client = OpenAIClient(api_key=api_key)

# result = client.create_embedding(text=text_8192, model="text-embedding-3-large")

# print(len(result))
