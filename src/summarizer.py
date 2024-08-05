"""Summarizer."""

import os

import pandas as pd

from client import OpenAIClient
from prompt import OpenAIPrompt


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

        self._summarized_df = (
            pd.read_parquet(self._output_path)
            if os.path.exists(self._output_path)
            else pd.DataFrame(columns=["title", "url", "content"])
        )
        self._unsummarized_df = self._read_docs()

        self._openai_client = OpenAIClient(api_key=api_key)
        self._openai_prompt = OpenAIPrompt(prompt_name=prompt_name)

    def _read_docs(self) -> pd.DataFrame:
        """Read the documentation files and return a DataFrame.

        :returns: DataFrame containing the documentation content.
        """
        dfs = []

        summarized_title = set(self._summarized_df["title"].tolist())

        def __read_docs(input_dir: str) -> pd.DataFrame:
            for entry in os.scandir(input_dir):
                if entry.is_dir() and entry.name in self._target_subdirs:
                    __read_docs(entry.path)
                elif entry.is_file():
                    df = pd.read_json(entry.path)

                    if df["title"].item() not in summarized_title:
                        dfs.append(df)

            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        return __read_docs(input_dir=self._input_dir)

    @staticmethod
    def _split_text(text: str, chunk_size: int) -> list[str]:
        """"""
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def summarize_docs(
        self,
        model: str,
        chunk_size: int,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ) -> None:
        """Summarize the documentation content and save the summaries to a parquet file.

        :param chunk_size: The size of text chunks for summarization.
        :param model: The model name to be used for summarization.
        :param temperature: The temperature setting for the model.
        """
        print(f"The number of rows in the DataFrame to summarize: {len(self._unsummarized_df)}")

        for idx, row in self._unsummarized_df.iterrows():
            splited_texts = self._split_text(text=row["content"], chunk_size=chunk_size)

            summary = []
            for text in splited_texts:
                parameters = {"text": text}
                messages = self._openai_prompt.format(parameters=parameters)

                response = self._openai_client.create_completion(
                    messages=messages, model=model, response_format=response_format, temperature=temperature
                )

                summary.append(response)

            self._unsummarized_df.loc[idx, "summary"] = " ".join(summary)

        summary_df = pd.concat([self._summarized_df, self._unsummarized_df], ignore_index=True)

        summary_df.to_parquet(path=self._output_path, index=False)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    target_subdirs = ["home"]  # , "setup", "test"]
    prompt_name = "summary"

    summarizer = DocsSummarizer(
        target_subdirs=target_subdirs,
        api_key=api_key,
        prompt_name=prompt_name,
    )

    model = "gpt-4o-mini"
    chunk_size = 128000
    summarizer.summarize_docs(model=model, chunk_size=chunk_size)
