"""Embedder."""

import os

import pandas as pd
import tiktoken

from client import OpenAIClient


class DocsEmbedder:
    """Documentations Embedder."""

    def __init__(self, api_key: str) -> None:
        self._openai_client = OpenAIClient(api_key=api_key)
        self._input_path = os.path.join(os.path.dirname(__file__), "..", "data/summary.parquet")
        self._output_path = os.path.join(os.path.dirname(__file__), "..", "data/vector_db.parquet")
        self._base_columns = [
            "title",
            "url",
            "content",
            "summary",
            "embedding_input",
            "embedding_output",
        ]

    @staticmethod
    def make_chunks(data: str | list[int], length: int) -> list[str]:
        """"""
        return [data[i : i + length] for i in range(0, len(data), length)]

    def _get_embedding_input(self, encoding_name: str, max_tokens: int) -> pd.DataFrame:
        """"""
        df = pd.read_parquet(self._input_path)
        df["embedding_input"] = df["title"] + " " + df["content"]

        encoding = tiktoken.get_encoding(encoding_name=encoding_name)

        rows = []
        for row in df.itertuples(index=False):
            embedding_input = row.embedding_input
            tokens = encoding.encode(text=embedding_input)

            for chunk in self.make_chunks(data=tokens, length=max_tokens):
                text = encoding.decode(tokens=chunk)

                rows.append(
                    {
                        "title": row.title,
                        "url": row.url,
                        "content": row.content,
                        "summary": row.summary,
                        "embedding_input": text,
                        "embedding_output": None,
                    },
                )

        return pd.DataFrame(data=rows, columns=self._base_columns)

    def _get_embedding_output(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """"""
        for idx, row in df.iterrows():
            embedding_output = self._openai_client.create_embedding(text=row["embedding_input"], model=model)
            df.loc[idx, "embedding_output"] = embedding_output

        return df

    def embed_docs(
        self,
        model: str = "text-embedding-3-large",
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8192,
    ) -> None:
        """"""
        embedding_df = self._get_embedding_input(encoding_name=encoding_name, max_tokens=max_tokens)
        embedding_df = self._get_embedding_output(df=embedding_df, model=model)

        embedding_df.to_parquet(path=self._output_path, index=False)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    embedder = DocsEmbedder(api_key=api_key)
    embedder.embed_docs()
