"""API main function."""

import os

from fastapi import Body, FastAPI, HTTPException
from rag import DocsRAG

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PROMPT_NAME = "helper"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192
COMPLETION_MODEL = "gpt-4o-mini"
COMPLETION_CONTEXT_WINDOW = 128000


rag = DocsRAG(api_key=OPENAI_API_KEY, prompt_name=PROMPT_NAME)
app = FastAPI()


@app.post("/chat")
async def chat(request: dict[str, str] = Body(...)) -> str:
    """Generate a chat response based on input text using the RAG system.

    :param request: A dictionary containing the input text for generating the response.
                    The dictionary must include a 'text' key with a string value.
    :returns: The generated chat response as a string.
    :raises HTTPException: If the 'text' field is missing from the request or any processing error occurs.

    Example:
        To use this endpoint, send a POST request with a JSON body containing the 'text' field:
        {
            "text": "What are the considerations for large Kubernetes clusters?"
        }
        The response will be a string generated based on the retrieved documents.
    """
    try:
        text = request.get("text")

        if not text:
            raise HTTPException(status_code=400, detail="'text' field is required.")

        rag.check_token_limit(
            text=text,
            embedding_model=EMBEDDING_MODEL,
            embedding_max_tokens=EMBEDDING_MAX_TOKENS,
            completion_model=COMPLETION_MODEL,
            completion_context_window=COMPLETION_CONTEXT_WINDOW,
        )

        search_df = rag.get_similarity_search(text=text, model=EMBEDDING_MODEL)

        return rag.create_chat_response(search_df=search_df, text=text, model=COMPLETION_MODEL)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clear")
async def clear() -> dict[str, str]:
    """Clear the current chat history.

    :returns: A message confirming that the chat history has been cleared.
    """
    rag.clear_chat_history()
    return {"message": "Chat history cleared."}


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint.

    :returns: A dictionary indicating the health status of the API.
    """
    return {"status": "healthy"}
