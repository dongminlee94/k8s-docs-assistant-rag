"""Embedder."""

import os

import pandas as pd
import tiktoken

from client import OpenAIClient


class DocsEmbedder:
    """Documentations Embedder.

    This class is used to embed documentation content using OpenAI's embedding models.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize the DocsEmbedder.

        :param api_key: The API key to authenticate with the OpenAI service.
        """
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
    def _make_chunks(data: str | list[int], length: int) -> list[str]:
        """
        Split the data into chunks of the specified length.

        :param data: The data to be split (either a string or a list of tokens).
        :param length: The length of each chunk.
        :returns: A list of chunks.

        Example:
            >>> data = "This is a sample text to be chunked."
            >>> chunks = DocsEmbedder.make_chunks(data=data, length=10)
            >>> print(chunks)
            ['This is a ', 'sample tex', 't to be c', 'hunked.']
        """
        return [data[i : i + length] for i in range(0, len(data), length)]

    def _get_embedding_input(self, model: str, max_tokens: int) -> pd.DataFrame:
        """
        Prepare the embedding input by combining title and content, and splitting into chunks.

        :param model: The name of the embedding model to use.
        :param max_tokens: The maximum number of tokens per chunk.
        :returns: A DataFrame with the embedding input.
        """
        df = pd.read_parquet(self._input_path)
        df["embedding_input"] = df["title"] + " " + df["content"]

        encoder = tiktoken.encoding_for_model(model_name=model)

        rows = []
        for row in df.itertuples():
            tokens = encoder.encode(text=row.embedding_input)

            for chunk in self._make_chunks(data=tokens, length=max_tokens):
                text = encoder.decode(tokens=chunk)

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
        """
        Generate embeddings for the embedding input using the specified model.

        :param df: DataFrame containing the embedding input.
        :param model: The name of the embedding model to use.
        :returns: DataFrame with the embedding output.
        """
        for row in df.itertuples():
            embedding_output = self._openai_client.create_embedding(text=row.embedding_input, model=model)
            df.loc[row.Index, "embedding_output"] = embedding_output

        return df

    def embed_docs(
        self,
        model: str,
        max_tokens: int,
    ) -> None:
        """
        Embed the documentation content and save the embeddings to a file.

        :param model: The name of the embedding model to use.
        :param max_tokens: The maximum number of tokens per chunk.
        :note: The resulting embeddings are saved to the specified output path.

        Example:
            >>> embedder = DocsEmbedder(api_key="your_api_key")
            >>> embedder.embed_docs(model="text-embedding-3-large", max_tokens=8192)
        """
        print("Starting the embedding process...")

        embedding_df = self._get_embedding_input(model=model, max_tokens=max_tokens)
        embedding_df = self._get_embedding_output(df=embedding_df, model=model)

        embedding_df.to_parquet(path=self._output_path, index=False)

        print(f"Embedding process completed. The results are saved to {self._output_path.split('/../')[-1]}.")


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    embedder = DocsEmbedder(api_key=api_key)

    model = "text-embedding-3-large"
    max_tokens = 8192

    embedder.embed_docs(model=model, max_tokens=max_tokens)
