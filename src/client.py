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
        self._client = OpenAI(api_key=api_key)

    def create_embedding(self, text: str, model: str) -> list[float]:
        """Create an embedding for the given text using the specified model.

        :param text: The input text to be embedded.
        :param model: The name of the model to be used for embedding.
        :returns: A list of floats representing the embedding of the input text.
        """
        return self._client.embeddings.create(input=text, model=model).data[0].embedding

    def create_completion(
        self,
        messages: str | list[dict[str, str]],
        model: str,
        response_format: dict[str, str] | None = None,
        temperature: int = 0,
    ) -> str | None:
        """Create a completion for the given messages using the specified model.

        :param messages: A single message or a list of message dictionaries to generate the completion.
        :param model: The name of the model to be used for generating the completion.
        :param response_format: Optional dictionary specifying the format of the response.
        :param temperature: The temperature value for sampling; higher values make the output more random.
        :returns: The content of the generated completion.
        :raises Exception: If the response does not contain exactly one choice.
        """
        response_format = response_format if response_format else {}

        response = self._client.chat.completions.create(
            messages=messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
        )

        if len(response.choices) != 1:
            raise Exception(response)

        return response.choices[0].message.content
