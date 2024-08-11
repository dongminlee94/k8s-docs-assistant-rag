"""Summarizer."""

import os

import pandas as pd
from core.client import OpenAIClient
from core.prompt import PromptTemplate
from core.util import make_chunks
from tqdm import tqdm


class DocsSummarizer:
    """Documentation Summarizer."""

    def __init__(
        self,
        target_subdirs: list[str],
        api_key: str,
        prompt_name: str,
    ) -> None:
        """Initialize the DocsSummarizer.

        :param target_subdirs: List of target subdirectories to process.
        :param api_key: The API key for accessing the model.
        :param prompt_name: The name of the prompt to use.
        """
        self._input_dir = os.path.join(os.path.dirname(__file__), "../../..", "data/docs")
        self._output_path = os.path.join(os.path.dirname(__file__), "../../..", "data/summary.parquet")
        self._target_subdirs = target_subdirs
        self._base_columns = ["title", "url", "content"]

        self._openai_client = OpenAIClient(api_key=api_key)
        self._prompt_template = PromptTemplate(prompt_name=prompt_name)

    def _read_docs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the documentation files and return a DataFrame.

        :returns: DataFrame containing the documentation content.
        """
        if os.path.exists(self._output_path):
            summarized_df = pd.read_parquet(self._output_path)
            summarized_urls = set(summarized_df["url"].unique())
        else:
            summarized_df = pd.DataFrame(columns=self._base_columns)
            summarized_urls = set()

        dfs = []

        def __read_docs(input_dir: str) -> pd.DataFrame:
            for entry in os.scandir(input_dir):
                if entry.is_dir() and entry.name in self._target_subdirs:
                    __read_docs(entry.path)
                elif entry.is_file():
                    df = pd.read_json(entry.path)

                    if df["url"].item() not in summarized_urls:
                        dfs.append(df)

            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        unsummarized_df = __read_docs(input_dir=self._input_dir)

        return summarized_df, unsummarized_df

    def summarize_docs(
        self,
        model: str,
        context_window: int,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
        verbose: bool = False,
    ) -> None:
        """Summarize the documentation files.

        :param model: The model to be used for generating summaries.
        :param context_window: The maximum number of tokens per request to the model.
        :param response_format: The format of the response from the model.
        :param temperature: The temperature value for sampling; higher values make the output more random.
        :param verbose: Whether to display progress.

        Example:
            >>> summarizer = DocsSummarizer(
            ...     target_subdirs=["home", "setup", "test"],
            ...     api_key="your_api_key",
            ...     prompt_name="summary"
            ... )
            >>> summarizer.summarize_docs(model="gpt-4o-mini", context_window=128000, verbose=True)
        """
        summarized_df, unsummarized_df = self._read_docs()

        print(f"The number of rows in the DataFrame to summarize: {len(unsummarized_df)}")

        for row in tqdm(unsummarized_df.itertuples()) if verbose else unsummarized_df.itertuples():
            splited_texts = make_chunks(data=row.content, length=context_window)

            summary = []
            for text in splited_texts:
                parameters = {"text": text}
                messages = self._prompt_template.format(parameters=parameters)

                response = self._openai_client.create_completion(
                    messages=messages, model=model, response_format=response_format, temperature=temperature
                )

                summary.append(response)

            unsummarized_df.loc[row.Index, "summary"] = " ".join(summary)

        summary_df = pd.concat([summarized_df, unsummarized_df], ignore_index=True)

        summary_df.to_parquet(path=self._output_path, index=False)
