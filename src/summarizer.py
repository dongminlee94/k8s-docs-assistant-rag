"""Summarizer."""

import os

import pandas as pd
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter

from prompt import LangchainPrompt


class DocsSummarizer:
    """Documentations Summarizer."""

    def __init__(
        self,
        target_subdirs: list[str],
        chunk_size: int,
        chunk_overlap: int,
        model: str,
        temperature: int,
        api_key: str,
        prompt_name: str,
    ) -> None:
        """Initialize the DocsSummarizer.

        :param target_subdirs: List of target subdirectories to process.
        :param chunk_size: The size of text chunks for summarization.
        :param chunk_overlap: The overlap between text chunks.
        :param model: The model name to be used for summarization.
        :param temperature: The temperature setting for the model.
        :param api_key: The API key for accessing the model.
        :param prompt_name: The name of the prompt to use.
        """
        self._input_dir = os.path.join(os.path.dirname(__file__), "..", "data/docs")
        self._output_path = os.path.join(os.path.dirname(__file__), "..", "data/vector_db.parquet")
        self._target_subdirs = target_subdirs

        self._summarized_df = pd.read_parquet(self._output_path)
        self._unsummarized_df = self._read_docs()

        self._text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._chat_model = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        self._prompt = LangchainPrompt(prompt_name=prompt_name)

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

    def summarize_docs(self) -> None:
        """Summarize the documentation content and save the summaries to a parquet file."""
        print(f"The number of rows in the DataFrame to summarize: {len(self._unsummarized_df)}")

        for idx, row in self._unsummarized_df.iterrows():
            splited_texts = self._text_splitter.split_text(row["content"])
            documented_texts = [Document(page_content=text) for text in splited_texts]

            summary = []
            for text in documented_texts:
                chain = load_summarize_chain(
                    llm=self._chat_model,
                    chain_type="stuff",
                    prompt=self._prompt.template,
                )

                parameters = {"input_documents": [text]}
                output = chain.invoke(parameters, return_only_outputs=True)

                summary.append(output["output_text"])

            self._unsummarized_df.loc[idx, "summary"] = " ".join(summary)

        summary_df = pd.concat([self._summarized_df, self._unsummarized_df], ignore_index=True)

        summary_df.to_parquet(path=self._output_path, index=False)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    # "concepts": 144개
    # "contribute": 32개
    # "home": 1개
    # "reference": 309개
    # "setup": 22개
    # "tasks": 197개
    # "test": 1개
    # "tutorials": 37개

    target_subdirs = ["home", "setup", "test"]
    chunk_size = 128000
    chunk_overlap = 0
    model = "gpt-4o-mini"
    temperature = 0
    api_key = api_key
    prompt_name = "summarizer"

    summarizer = DocsSummarizer(
        target_subdirs=target_subdirs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        temperature=temperature,
        api_key=api_key,
        prompt_name=prompt_name,
    )

    summarizer.summarize_docs()
