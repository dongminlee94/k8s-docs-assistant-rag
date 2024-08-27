"""Embedder."""

import os

import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm


class DocsEmbedder:
    """Documentation Embedder.

    This class is used to embed documentation content using OpenAI's embedding models.
    """

    def __init__(self, api_key: str, target_subdomains: list[str]) -> None:
        """
        Initialize the DocsEmbedder.

        :param api_key: The API key to authenticate with the OpenAI service.
        :param target_subdomains: List of target subdomains to process.
        """
        self._openai_client = OpenAI(api_key=api_key)
        self._target_subdomains = target_subdomains

        self._input_dir = os.path.join(os.path.dirname(__file__), "../..", "data/docs")
        self._output_path = os.path.join(os.path.dirname(__file__), "../..", "data/vector_db.parquet")

        self._base_columns = [
            "file_path",
            "title",
            "url",
            "content",
            "embedding_input",
            "embedding_output",
        ]

    def _get_vector_db_df(self) -> pd.DataFrame:
        """Load the existing vector database from a Parquet file, or initialize an empty DataFrame.

        :returns: A DataFrame containing the existing vector database or an empty DataFrame with base columns.
        """
        if os.path.exists(self._output_path):
            vector_db_df = pd.read_parquet(self._output_path)
        else:
            vector_db_df = pd.DataFrame(columns=self._base_columns)

        return vector_db_df

    def _read_docs(self) -> pd.DataFrame:
        """Read and filter documentation files from the target subdomains.

        :returns: A DataFrame containing the unembedded documentation data.
        :note: This method reads all JSON files from the specified input directory, filters out the ones that
               have already been embedded (based on their URLs), and returns the remaining data.
        """
        vector_db_df = self._get_vector_db_df()
        vector_db_urls = set(vector_db_df["url"])

        dfs = []

        def __read_docs(input_dir: str) -> pd.DataFrame:
            for entry in os.scandir(input_dir):
                if entry.is_dir() and entry.name in self._target_subdomains:
                    __read_docs(entry.path)
                elif entry.is_file():
                    df = pd.read_json(entry.path)
                    url = df.iloc[0]["url"]

                    if url in vector_db_urls:
                        content = vector_db_df[vector_db_df["url"] == url].iloc[0]["content"]

                        if content != df.iloc[0]["content"]:
                            dfs.append(df)
                    else:
                        dfs.append(df)

            return pd.concat(objs=dfs, ignore_index=True) if dfs else pd.DataFrame(columns=self._base_columns)

        return __read_docs(input_dir=self._input_dir)

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

    def _get_embedding_input(self, model: str, max_tokens: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the embedding input by combining title and content, and splitting into chunks.

        :param model: The name of the embedding model to use.
        :param max_tokens: The maximum number of tokens per chunk.
        :returns: A tuple containing two DataFrames:
            - embedded_df: DataFrame with already embedded data.
            - unembedded_df: DataFrame with data to be embedded.
        """
        vector_db_df = self._get_vector_db_df()

        docs_df = self._read_docs()
        docs_df["embedding_input"] = docs_df["title"] + " " + docs_df["content"]

        embedded_df = vector_db_df[~vector_db_df["url"].isin(docs_df["url"].unique())]
        embedded_urls = set(embedded_df["url"])

        encoder = tiktoken.encoding_for_model(model_name=model)

        rows = []
        for row in docs_df.itertuples():
            if row.url in embedded_urls:
                continue

            tokens = encoder.encode(text=row.embedding_input)

            for chunk in self._make_chunks(data=tokens, length=max_tokens):
                text = encoder.decode(tokens=chunk)

                rows.append(
                    {
                        "file_path": row.file_path,
                        "title": row.title,
                        "url": row.url,
                        "content": row.content,
                        "embedding_input": text,
                        "embedding_output": None,
                    },
                )

        return embedded_df, pd.DataFrame(data=rows, columns=self._base_columns)

    def _get_embedding_output(
        self,
        model: str,
        unembedded_df: pd.DataFrame,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generate embeddings for the unembedded input using the specified model.

        :param model: The name of the embedding model to use.
        :param unembedded_df: DataFrame containing data to be embedded.
        :param verbose: Whether to display progress.
        :returns: A DataFrame with the new embeddings.
        """
        print(f"The number of rows in the DataFrame to embed: {len(unembedded_df)}")

        for row in tqdm(unembedded_df.itertuples()) if verbose else unembedded_df.itertuples():
            embedding_output = (
                self._openai_client.embeddings.create(input=row.embedding_input, model=model)
                .data[0]
                .embedding
            )
            unembedded_df.loc[row.Index, "embedding_output"] = embedding_output

        return unembedded_df

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
        new_embedded_df = self._get_embedding_output(
            unembedded_df=unembedded_df, model=model, verbose=verbose
        )

        if not new_embedded_df.empty:
            print(
                f'The number of newly embedded file paths: {len(new_embedded_df["file_path"])}\n'
                f'The list of newly embedded file paths: {new_embedded_df["file_path"].tolist()}\n'
            )

        vector_db_df = pd.concat([embedded_df, new_embedded_df], ignore_index=True)
        vector_db_df.to_parquet(path=self._output_path, index=False)
