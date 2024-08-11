"""Retrieval-Augmented Generation (RAG)."""

import os

import faiss
import numpy as np
import pandas as pd
import tiktoken
from client import OpenAIClient

from prompt import PromptTemplate


class DocsRAG:
    """Documentation Retrieval-Augmented Generation (DocsRAG).

    This class implements the RAG technique by retrieving relevant documents from a vector database
    and generating responses using a language model based on the retrieved content.
    """

    def __init__(self, api_key: str, prompt_name: str) -> None:
        """Initialize the DocsRAG class.

        :param api_key: The API key to authenticate with the OpenAI service.
        :param prompt_name: The name of the prompt template to be used for generating responses.
        """
        self._openai_client = OpenAIClient(api_key=api_key)
        self._prompt_template = PromptTemplate(prompt_name=prompt_name)

        self._vector_db = pd.read_parquet(
            os.path.join(os.path.dirname(__file__), "../../..", "data/vector_db.parquet")
        )
        self._index = self._get_index()

    def _get_index(self) -> faiss.IndexIDMap:
        """Create and return the Faiss index for efficient similarity search.

        :returns: A Faiss IndexIDMap object for performing similarity search on the vector database.
        """
        embedding_data = np.array(list(self._vector_db["embedding_output"]))

        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_data.shape[1]))
        index.add_with_ids(embedding_data, np.array(range(0, len(self._vector_db))))

        return index

    def check_text_limit(
        self,
        text: str,
        embedding_model: str,
        embedding_max_tokens: int,
        completion_model: str,
        completion_context_window: int,
    ) -> None:
        """Check if the input text is within the token limits of the models.

        :param text: The input text to check.
        :param embedding_model: The embedding model name to check against its token limit.
        :param embedding_max_tokens: The maximum number of tokens allowed for the embedding model.
        :param completion_model: The completion model name to check against its context window.
        :param completion_context_window: The max tokens allowed in the completion model's context window.
        :raises ValueError: If the text exceeds the token limit for either model.

        Example:
            >>> rag = DocsRAG(api_key="your_api_key", prompt_name="helper")
            >>> text = "What are the considerations for large Kubernetes clusters?"
            >>> embedding_model = "text-embedding-3-large"
            >>> embedding_max_tokens = 8192
            >>> completion_model = "gpt-4o-mini"
            >>> completion_context_window = 128000
            >>> rag.check_text_limit(
            ...     text=text,
            ...     embedding_model=embedding_model,
            ...     embedding_max_tokens=embedding_max_tokens,
            ...     completion_model=completion_model,
            ...     completion_context_window=completion_context_window,
            ... )
        """
        for model, limit in [
            [embedding_model, embedding_max_tokens],
            [completion_model, completion_context_window],
        ]:
            encoder = tiktoken.encoding_for_model(model_name=model)
            tokens = encoder.encode(text=text)

            if len(tokens) > limit:
                raise ValueError(
                    f"Input text exceeds the token limit for {model}. Limit: {limit}, Tokens: {len(tokens)}"
                )

    def get_similarity_search(self, text: str, model: str, top_k: int = 5) -> pd.DataFrame:
        """Perform a similarity search on the vector database.

        :param text: The input text to search for similar documents.
        :param model: The name of the embedding model to use.
        :param top_k: The number of top similar documents to retrieve.
        :returns: A DataFrame containing the top_k most similar documents.

        Example:
            >>> rag = DocsRAG(api_key="your_api_key", prompt_name="helper")
            >>> text = "What are the considerations for large Kubernetes clusters?"
            >>> model = "text-embedding-3-large"
            >>> search_df = rag.get_similarity_search(text=text, model=model)
            >>> print(search_df)
            # DataFrame with columns containing the top_k similar documents.
        """
        embedding_output = self._openai_client.create_embedding(text=text, model=model)
        search_result = self._index.search(np.array([embedding_output]), top_k)

        return self._vector_db.iloc[list(search_result[1][0])].reset_index(drop=True)

    def create_chat_response(
        self,
        search_df: pd.DataFrame,
        text: str,
        model: str,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ):
        """Generate a response using the language model based on the retrieved documents.

        :param search_df: A DataFrame containing the retrieved documents.
        :param text: The input query or text to respond to.
        :param model: The name of the model to use for generating the response.
        :param response_format: Optional dictionary specifying the format of the response.
        :param temperature: Setting for response variability; higher values increase randomness.
        :returns: A string containing the generated response.

        Example:
            >>> rag = DocsRAG(api_key="your_api_key", prompt_name="helper")
            >>> text = "What are the considerations for large Kubernetes clusters?"
            >>> model = "gpt-4o-mini"
            >>> search_df = rag.get_similarity_search(text=text, model="text-embedding-3-large")
            >>> response = rag.create_chat_response(search_df=search_df, text=text, model=model)
            >>> print(response)
            # "When managing large Kubernetes clusters, consider..."
        """
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
    rag = DocsRAG(api_key=api_key, prompt_name=prompt_name)

    text = "쿠버네티스에서 대형 클러스터에 대한 고려 사항으로는 어떤 것들이 있니?"
    # text = "카프카에 대해 알려줘."
    # text = "쿠버네티스가 뭐야?"
    # text = "바보가 뭐야?"

    embedding_model = "text-embedding-3-large"
    embedding_max_tokens = 8192
    completion_model = "gpt-4o-mini"
    completion_context_window = 128000

    rag.check_text_limit(
        text=text,
        embedding_model=embedding_model,
        embedding_max_tokens=embedding_max_tokens,
        completion_model=completion_model,
        completion_context_window=completion_context_window,
    )

    search_df = rag.get_similarity_search(text=text, model=embedding_model)

    response = rag.create_chat_response(
        search_df=search_df, text=text, model=completion_model, context_window=completion_context_window
    )

    print(response)
