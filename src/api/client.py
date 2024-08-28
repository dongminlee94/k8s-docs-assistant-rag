"""Client."""

from openai import OpenAI


class OpenAIClient:
    """OpenAI Client.

    This class provides an interface to interact with OpenAI API services such as embeddings and completions.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the OpenAIClient.

        :param api_key: The API key to authenticate with the OpenAI service.
        """
        # Initialize the OpenAI client with the provided API key
        self._client = OpenAI(api_key=api_key)

    def create_embedding(self, text: str, model: str) -> list[float]:
        """Create an embedding for the given text using the specified model.

        :param text: The input text to be embedded.
        :param model: The name of the model to be used for embedding.
        :returns: A list of floats representing the embedding of the input text.

        Example:
            >>> client = OpenAIClient(api_key="your_api_key")
            >>> text = "This is a sample text to embed."
            >>> model = "text-embedding-3-large"
            >>> embedding = client.create_embedding(text=text, model=model)
            >>> print(embedding)
            [0.12345, 0.67890, ...]  # List of float values representing the embedding
        """
        return self._client.embeddings.create(input=text, model=model).data[0].embedding

    def create_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        response_format: dict[str, str] | None = None,
        temperature: int = 0,
    ) -> str:
        """Create a completion for the given messages using the specified model.

        :param messages: A single message or a list of message dictionaries to generate the completion.
        :param model: The name of the model to be used for generating the completion.
        :param response_format: Optional dictionary specifying the format of the response.
        :param temperature: The temperature value for sampling; higher values make the output more random.
        :returns: The content of the generated completion.
        :raises Exception: If the response does not contain exactly one choice.

        Example:
            >>> client = OpenAIClient(api_key="your_api_key")
            >>> messages = [
            ...     {"role": "system", "content": "You are an assistant."},
            ...     {"role": "user", "content": "Tell me a joke."}
            ... ]
            >>> model = "gpt-4o-mini"
            >>> response = client.create_completion(messages=messages, model=model, temperature=0.7)
            >>> print(response)
            "Why don't scientists trust atoms? Because they make up everything!"
        """
        # Set default response format if none is provided
        response_format = response_format if response_format else {}

        # Use OpenAI client to generate a completion for the given messages using the specified model
        response = self._client.chat.completions.create(
            messages=messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
        )

        # Check if the response contains exactly one choice
        if len(response.choices) != 1:
            raise Exception(response)

        return response.choices[0].message.content
