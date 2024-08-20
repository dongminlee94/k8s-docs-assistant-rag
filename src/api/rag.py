"""Retrieval-Augmented Generation (RAG)."""

import os
import time

import faiss
import numpy as np
import pandas as pd
import tiktoken
import yaml
from client import OpenAIClient


class DocsRAG:
    """Documentation Retrieval-Augmented Generation (DocsRAG).

    This class implements the RAG technique by retrieving relevant documents from a vector database
    and generating responses using a language model based on the retrieved content.
    """

    def __init__(
        self,
        api_key: str,
        prompt_name: str,
        embedding_model: str,
        embedding_max_tokens: int,
        completion_model: str,
        completion_context_window: int,
    ) -> None:
        """Initialize the DocsRAG class.

        :param api_key: The API key to authenticate with the OpenAI service.
        :param prompt_name: The name of the prompt template to be used for generating responses.
        :param embedding_model: The embedding model name to check against its token limit.
        :param embedding_max_tokens: The maximum number of tokens allowed for the embedding model.
        :param completion_model: The completion model name to check against its context window.
        :param completion_context_window: The max tokens allowed in the completion model's context window.
        """
        self._openai_client = OpenAIClient(api_key=api_key)
        self._prompt_name = prompt_name
        self._embedding_model = embedding_model
        self._embedding_max_tokens = embedding_max_tokens
        self._completion_model = completion_model
        self._completion_context_window = completion_context_window

        self._vector_db = self._get_vector_db()
        self._index = self._get_index()

        self.clear_chat_history()

    def _get_vector_db(self) -> pd.DataFrame:
        """Retrieve the vector database from a parquet file.

        :returns: A pandas DataFrame containing the vector database.
        """
        file_path = os.path.join(os.path.dirname(__file__), "../..", "data/vector_db.parquet")

        while not os.path.exists(file_path):
            print("The vector_db.parquet file is not found. Please wait while it is being created...")
            time.sleep(10)

        return pd.read_parquet(file_path)

    def _get_index(self) -> faiss.IndexIDMap:
        """Create and return the Faiss index for efficient similarity search.

        :returns: A Faiss IndexIDMap object for performing similarity search on the vector database.
        """
        embedding_data = np.array(list(self._vector_db["embedding_output"]))

        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_data.shape[1]))
        index.add_with_ids(embedding_data, np.array(range(0, len(self._vector_db))))

        return index

    def clear_chat_history(self) -> None:
        """
        Clear the current chat history and reset it to the initial base prompt.

        Example:
            >>> rag = DocsRAG(
            ...     api_key="your_api_key",
            ...     prompt_name="helper",
            ...     embedding_model="text-embedding-3-large",
            ...     embedding_max_tokens=8192,
            ...     completion_model="gpt-4o-mini",
            ...     completion_context_window=128000
            ... )
            >>> rag.clear_chat_history()
            >>> print(rag.chat)
            # Output: [{'role': 'system', 'content': 'You are an assistant.'}, ...]

            >>> rag.chat.append({"role": "user", "content": "쿠버네티스란 무엇인가요?"})
            >>> print(rag.chat)
            # Output: [{'role': 'system', 'content': 'You are an assistant.'},
            #          {'role': 'user', 'content': '쿠버네티스란 무엇인가요?'}]

            >>> rag.clear_chat_history()
            >>> print(rag.chat)
            # Output: [{'role': 'system', 'content': 'You are an assistant.'}, ...] # Reset to initial state
        """
        prompt_file_path = os.path.join(
            os.path.dirname(__file__), "../..", "prompt", f"{self._prompt_name}.yaml"
        )

        with open(prompt_file_path, "r", encoding="utf-8") as file:
            prompt = yaml.safe_load(file)

        if not isinstance(prompt, list) or not all(
            isinstance(piece, str) and len(msg) == 2 for msg in prompt for piece in msg
        ):
            raise ValueError('The prompt must be a list of pairs like [["role", "content"]].')

        self.chat = [{"role": msg[0], "content": msg[1]} for msg in prompt]

    def check_token_limit(self, content: str) -> bool:
        """Check if the current chat history and the new content exceed the token limit for the models.

        :param content: The new content to be added to the chat history.
        :return: `True` if the token limit is within acceptable range, `False` if exceeded.

        Example:
        >>> rag = DocsRAG(
        ...     api_key="your_api_key",
        ...     prompt_name="helper",
        ...     embedding_model="text-embedding-3-large",
        ...     embedding_max_tokens=8192,
        ...     completion_model="gpt-4o-mini",
        ...     completion_context_window=128000
        ... )
        >>> content = "What are the considerations for large Kubernetes clusters?"
        >>> vaild = rag.check_token_limit(content=content)
        >>> if vaild:
        ...     print("Token limit is within acceptable range.")
        ... else:
        ...     print("Token limit exceeded.")
        """
        contents = "".join([chat["content"] for chat in self.chat])
        contents += content

        for model, limit, text in [
            [self._embedding_model, self._embedding_max_tokens, content],
            [self._completion_model, self._completion_context_window, contents],
        ]:
            encoder = tiktoken.encoding_for_model(model_name=model)
            tokens = encoder.encode(text=text)

            if len(tokens) > limit:
                return False

        return True

    def get_similarity_search(self, content: str, top_k: int = 5) -> pd.DataFrame:
        """Perform a similarity search on the vector database.

        :param content: The input content to search for similar documents.
        :param top_k: The number of top similar documents to retrieve.
        :returns: A DataFrame containing the top_k most similar documents.

        Example:
            >>> rag = DocsRAG(
            ...     api_key="your_api_key",
            ...     prompt_name="helper",
            ...     embedding_model="text-embedding-3-large",
            ...     embedding_max_tokens=8192,
            ...     completion_model="gpt-4o-mini",
            ...     completion_context_window=128000
            ... )
            >>> content = "What are the considerations for large Kubernetes clusters?"
            >>> search_df = rag.get_similarity_search(content=content)
            >>> print(search_df)
            # DataFrame with columns containing the top_k similar documents.
        """
        embedding_output = self._openai_client.create_embedding(text=content, model=self._embedding_model)
        search_result = self._index.search(np.array([embedding_output]), top_k)

        return self._vector_db.iloc[list(search_result[1][0])].reset_index(drop=True)

    def create_chat_response(
        self,
        search_df: pd.DataFrame,
        content: str,
        response_format: dict[str, str] = {"type": "text"},
        temperature: int = 0,
    ) -> str:
        """Generate a response using the language model based on the retrieved documents.

        :param search_df: A DataFrame containing the retrieved documents.
        :param content: The input content to respond to.
        :param response_format: Optional dictionary specifying the format of the response.
        :param temperature: Setting for response variability; higher values increase randomness.
        :returns: A string containing the generated response.

        Example:
            >>> rag = DocsRAG(
            ...     api_key="your_api_key",
            ...     prompt_name="helper",
            ...     embedding_model="text-embedding-3-large",
            ...     embedding_max_tokens=8192,
            ...     completion_model="gpt-4o-mini",
            ...     completion_context_window=128000
            ... )
            >>> content = "What are the considerations for large Kubernetes clusters?"
            >>> search_df = rag.get_similarity_search(content=content, model="text-embedding-3-large")
            >>> response = rag.create_chat_response(search_df=search_df, content=content)
            >>> print(response)
            # "When managing large Kubernetes clusters, consider..."
        """
        search_df.drop_duplicates(subset=["title", "url", "content"], keep="first", inplace=True)

        documents = "--- RELATED INTERNAL DOCUMENTS ---\n"
        for row in search_df.itertuples():
            documents += f"관련된 내부 문서 {row.Index + 1}\n"
            documents += f" - Title: {row.title}\n"
            documents += f" - URL: {row.url}\n"
            documents += f" - Content: {row.content}\n\n"

        self.chat.append({"role": "system", "content": documents})
        self.chat.append({"role": "user", "content": f"--- QUESTION ---\n{content}"})

        response = self._openai_client.create_completion(
            messages=self.chat,
            model=self._completion_model,
            response_format=response_format,
            temperature=temperature,
        )

        self.chat.append({"role": "assistant", "content": response})

        return response
