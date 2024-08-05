"""Summarizer."""

import os

import pandas as pd

from client import OpenAIClient
from prompt import PromptTemplate


class DocsSummarizer:
    """Documentations Summarizer."""

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
        self._input_dir = os.path.join(os.path.dirname(__file__), "..", "data/docs")
        self._output_path = os.path.join(os.path.dirname(__file__), "..", "data/summary.parquet")
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
            summarized_title = set(summarized_df["title"].tolist())
        else:
            summarized_df = pd.DataFrame(columns=self._base_columns)
            summarized_title = []

        dfs = []

        def __read_docs(input_dir: str) -> pd.DataFrame:
            for entry in os.scandir(input_dir):
                if entry.is_dir() and entry.name in self._target_subdirs:
                    __read_docs(entry.path)
                elif entry.is_file():
                    df = pd.read_json(entry.path)

                    if df["title"].item() not in summarized_title:
                        dfs.append(df)

            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        unsummarized_df = __read_docs(input_dir=self._input_dir)

        return summarized_df, unsummarized_df

    @staticmethod
    def split_text(text: str, chunk_size: int) -> list[str]:
        """Split the text into chunks of specified size.

        :param text: The text to be split.
        :param chunk_size: The size of each chunk.
        :returns: A list of text chunks.
        """
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def summarize_docs(
        self,
        model: str,
        chunk_size: int,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ) -> None:
        """Summarize the documentation files.

        :param model: The model to be used for generating summaries.
        :param chunk_size: The size of each text chunk.
        :param response_format: The format of the response from the model.
        :param temperature: The temperature value for sampling; higher values make the output more random.

        Example:
            >>> summarizer = DocsSummarizer(
            ...     target_subdirs=["home", "setup", "test"],
            ...     api_key="your_api_key",
            ...     prompt_name="summary"
            ... )
            >>> summarizer.summarize_docs(model="gpt-4o-mini", chunk_size=128000)
        """
        summarized_df, unsummarized_df = self._read_docs()

        print(f"The number of rows in the DataFrame to summarize: {len(unsummarized_df)}")

        for idx, row in unsummarized_df.iterrows():
            splited_texts = self.split_text(text=row["content"], chunk_size=chunk_size)

            summary = []
            for text in splited_texts:
                parameters = {"text": text}
                messages = self._prompt_template.format(parameters=parameters)

                response = self._openai_client.create_completion(
                    messages=messages, model=model, response_format=response_format, temperature=temperature
                )

                summary.append(response)

            unsummarized_df.loc[idx, "summary"] = " ".join(summary)

        summary_df = pd.concat([summarized_df, unsummarized_df], ignore_index=True)

        summary_df.to_parquet(path=self._output_path, index=False)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    target_subdirs = ["home", "setup", "test"]
    prompt_name = "summary"

    summarizer = DocsSummarizer(
        target_subdirs=target_subdirs,
        api_key=api_key,
        prompt_name=prompt_name,
    )

    model = "gpt-4o-mini"
    chunk_size = 128000
    summarizer.summarize_docs(model=model, chunk_size=chunk_size)
