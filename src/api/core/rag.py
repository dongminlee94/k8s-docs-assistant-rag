"""Retrieval-Augmented Generation (RAG)."""

import os

import faiss
import numpy as np
import pandas as pd
import tiktoken
from client import OpenAIClient

from prompt import PromptTemplate


class RAG:
    """Retrieval-Augmented Generation (RAG)."""

    def __init__(self, api_key: str, prompt_name: str) -> None:
        """"""
        self._openai_client = OpenAIClient(api_key=api_key)
        self._prompt_template = PromptTemplate(prompt_name=prompt_name)

        self._vector_db = pd.read_parquet(
            os.path.join(os.path.dirname(__file__), "../../..", "data/vector_db.parquet")
        )
        self._index = self._get_index()

    def _get_index(self) -> faiss.IndexIDMap:
        """"""
        embedding = np.array(list(self._vector_db["embedding_output"]))

        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding.shape[1]))
        index.add_with_ids(embedding, np.array(range(0, len(self._vector_db))))

        return index

    def get_similarity_search(self, text: str, model: str, top_k: int = 5) -> pd.DataFrame:
        """"""
        embedding_output = self._openai_client.create_embedding(text=text, model=model)
        search_result = self._index.search(np.array([embedding_output]), top_k)

        return self._vector_db.iloc[list(search_result[1][0])].reset_index(drop=True)

    def create_chat_response(
        self,
        search_df: pd.DataFrame,
        text: str,
        model: str,
        context_window: int,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ):
        """"""
        encoder = tiktoken.encoding_for_model(model_name=model)
        tokens = encoder.encode(text=text)

        if len(tokens) > context_window:
            print("")
            raise

        search_df.drop_duplicates(subset=["title", "url", "content"], keep="first", inplace=True)

        documents = ""
        for row in search_df.itertuples():
            documents += f"관련된 내부 문서 {row.Index + 1}\n"
            documents += f" - Title: {row.title}\n"
            documents += f" - URL: {row.url}\n"
            documents += f" - Content: {row.content}\n\n"

        parameters = {"documents": documents, "question": text}
        messages = self._prompt_template.format(parameters=parameters)

        response = self._openai_client.create_completion(
            messages=messages, model=model, response_format=response_format, temperature=temperature
        )

        return response


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../../..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    prompt_name = "helper"
    rag = RAG(api_key=api_key, prompt_name=prompt_name)

    # text = "쿠버네티스에서 대형 클러스터에 대한 고려 사항으로는 어떤 것들이 있니?"
    # text = "카프카에 대해 알려줘."
    # text = "쿠버네티스가 뭐야?"
    text = "바보가 뭐야?"
    model = "text-embedding-3-large"

    search_df = rag.get_similarity_search(text=text, model=model)

    model = "gpt-4o-mini"
    context_window = 128000

    response = rag.create_chat_response(
        search_df=search_df, text=text, model=model, context_window=context_window
    )

    print(response)
