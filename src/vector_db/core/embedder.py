"""Embedder."""

import os

import pandas as pd
import tiktoken
from core.client import OpenAIClient
from core.util import make_chunks
from tqdm import tqdm


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
        self._input_path = os.path.join(os.path.dirname(__file__), "../../..", "data/summary.parquet")
        self._output_path = os.path.join(os.path.dirname(__file__), "../../..", "data/vector_db.parquet")
        self._base_columns = [
            "title",
            "url",
            "content",
            "summary",
            "embedding_input",
            "embedding_output",
        ]

    def _get_embedding_input(self, model: str, max_tokens: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the embedding input by combining title and content, and splitting into chunks.

        :param model: The name of the embedding model to use.
        :param max_tokens: The maximum number of tokens per chunk.
        :returns: A tuple containing two DataFrames:
            - embedded_df: DataFrame with already embedded data.
            - unembedded_df: DataFrame with data to be embedded.
        """
        if os.path.exists(self._output_path):
            embedded_df = pd.read_parquet(self._output_path)
            embedded_urls = set(embedded_df["url"].unique())
        else:
            embedded_df = pd.DataFrame(columns=self._base_columns[:-2])
            embedded_urls = set()

        unembedded_df = pd.read_parquet(self._input_path)
        unembedded_df["embedding_input"] = unembedded_df["title"] + " " + unembedded_df["content"]

        encoder = tiktoken.encoding_for_model(model_name=model)

        rows = []
        for row in unembedded_df.itertuples():
            if row.url in embedded_urls:
                continue

            tokens = encoder.encode(text=row.embedding_input)

            for chunk in make_chunks(data=tokens, length=max_tokens):
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

        return embedded_df, pd.DataFrame(data=rows, columns=self._base_columns)

    def _get_embedding_output(
        self,
        model: str,
        embedded_df: pd.DataFrame,
        unembedded_df: pd.DataFrame,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Generate embeddings for the unembedded input using the specified model and combine with embedded data.

        :param model: The name of the embedding model to use.
        :param embedded_df: DataFrame containing already embedded data.
        :param unembedded_df: DataFrame containing data to be embedded.
        :param verbose: Whether to display progress.
        :returns: A combined DataFrame with the new embeddings added to the existing embeddings.
        """
        print(f"The number of rows in the DataFrame to embed: {len(unembedded_df)}")

        for row in tqdm(unembedded_df.itertuples()) if verbose else unembedded_df.itertuples():
            embedding_output = self._openai_client.create_embedding(text=row.embedding_input, model=model)
            unembedded_df.loc[row.Index, "embedding_output"] = embedding_output

        return pd.concat([embedded_df, unembedded_df], ignore_index=True)

    def embed_docs(
        self,
        model: str,
        max_tokens: int,
        verbose: bool = False,
    ) -> None:
        """
        Embed the documentation content and save the embeddings to a file.

        :param model: The name of the embedding model to use.
        :param max_tokens: The maximum number of tokens per chunk.
        :param verbose: Whether to display progress.
        :note: The resulting embeddings are saved to the specified output path.

        Example:
            >>> embedder = DocsEmbedder(api_key="your_api_key")
            >>> embedder.embed_docs(model="text-embedding-3-large", max_tokens=8192, verbose=True)
        """
        embedded_df, unembedded_df = self._get_embedding_input(model=model, max_tokens=max_tokens)
        vector_db_df = self._get_embedding_output(
            embedded_df=embedded_df, unembedded_df=unembedded_df, model=model, verbose=verbose
        )

        vector_db_df.to_parquet(path=self._output_path, index=False)