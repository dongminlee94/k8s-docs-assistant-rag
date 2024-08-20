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


rag = DocsRAG(
    api_key=OPENAI_API_KEY,
    prompt_name=PROMPT_NAME,
    embedding_model=EMBEDDING_MODEL,
    embedding_max_tokens=EMBEDDING_MAX_TOKENS,
    completion_model=COMPLETION_MODEL,
    completion_context_window=COMPLETION_CONTEXT_WINDOW,
)
app = FastAPI()


@app.post("/chat")
async def chat(request: dict[str, str] = Body(...)) -> str:
    """Generate a chat response based on input content using the RAG system.

    :param request: A dictionary containing the input content for generating the response.
                    The dictionary must include a 'content' key with a string value.
    :returns: The generated chat response as a string.

    Example:
        To use this endpoint, send a POST request with a JSON body containing the 'content' field:
        {
            "content": "What are the considerations for large Kubernetes clusters?"
        }
        The response will be a string generated based on the retrieved documents.
    """
    try:
        content = request.get("content")

        if not content:
            raise HTTPException(status_code=400, detail='"content" field is required.')

        result = rag.check_token_limit(content=content)

        if result:
            detail = (
                f'Chat history or input exceeds the token limit for {result["model"]}. '
                f'Limit: {result["limit"]}, Token Length: {result["token_length"]} '
                f'Please press the "Clear" button or reduce the input text length.'
            )
            raise HTTPException(status_code=400, detail=detail)

        search_df = rag.get_similarity_search(content=content)
        response = rag.create_chat_response(search_df=search_df, content=content)

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clear")
async def clear() -> None:
    """Clear the current chat history."""
    rag.clear_chat_history()


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint.

    :returns: A dictionary indicating the health status of the API.
    """
    return {"status": "healthy"}
