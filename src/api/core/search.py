"""Search."""

import os

import faiss
import numpy as np
import pandas as pd
from client import OpenAIClient


class SimilaritySearch:
    """Similarity Search."""

    def __init__(self, api_key: str) -> None:
        """"""
        self._openai_client = OpenAIClient(api_key=api_key)
        self._vector_db = pd.read_parquet(
            os.path.join(os.path.dirname(__file__), "../../..", "data/vector_db.parquet")
        )
        self._index = self._get_index()

    def _get_index(self) -> faiss.IndexIDMap:
        """"""
        embedding = np.array(self._vector_db["embedding_output"].to_list())

        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding.shape[1]))
        index.add_with_ids(embedding, np.array(range(0, len(self._vector_db))))

        return index

    def get_similarity_search(self, text: str, model: str, top_k: int = 20) -> pd.DataFrame:
        """"""
        embedding_output = self._openai_client.create_embedding(text=text, model=model)
        search_result = self._index.search(np.array([embedding_output]), top_k)

        return self._vector_db.iloc[list(search_result[1][0])].reset_index(drop=True)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../../..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    search = SimilaritySearch(api_key=api_key)

    text = "쿠버네티스에서 대형 클러스터에 대한 고려 사항으로는 어떤 것들이 있니?"
    model = "text-embedding-3-large"

    search_df = search.get_similarity_search(text=text, model=model)

    print(search_df.head())
