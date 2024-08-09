"""Prompt."""

import os

import yaml


class PromptTemplate:
    """Prompt Template.

    This class is used to load and manage prompt templates.
    """

    def __init__(self, prompt_name: str) -> None:
        """Initialize the Prompt Template.

        :param prompt_name: The name of the prompt to load.
        """
        self._prompt_name = prompt_name
        self._prompt_template = self._get_template()

    def _get_template(self) -> list[dict[str, str]]:
        """Load the prompt template from a YAML file.

        :returns: A list of dictionaries with roles and templates.
        :raises ValueError: If the template format is not as expected.
        """
        prompt_file_path = os.path.join(
            os.path.dirname(__file__), "../../..", "prompt", f"{self._prompt_name}.yaml"
        )

        with open(prompt_file_path, "r", encoding="utf-8") as file:
            prompt = yaml.safe_load(file)

        template = prompt["template"]

        if isinstance(template, list) and all(
            isinstance(piece, str) and len(chat) == 2 for chat in template for piece in chat
        ):
            return [{chat[0]: chat[1]} for chat in template]

        raise ValueError("The template must be a list of pairs like [['role', 'content']].")

    def format(self, parameters: dict[str, str] | None = None) -> list[dict[str, str]]:
        """Format the prompt template with given parameters.

        :param parameters: A dictionary of parameters to format the template.
        :returns: A list of formatted messages with roles and contents.

        Example:
            >>> prompt_template = PromptTemplate(prompt_name="summary")
            >>> parameters = {"text": "Hello?"}
            >>> messages = prompt_template.format(parameters=parameters)
            >>> print(messages)
            [{'role': 'system', 'content': 'You are an assistant.'}, {'role': 'user', 'content': 'Hello?'}]
        """
        parameters = parameters if parameters else {}

        return [
            {"role": role, "content": template.format(**parameters)}
            for message in self._prompt_template
            for role, template in message.items()
        ]
