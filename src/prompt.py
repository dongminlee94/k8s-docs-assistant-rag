"""Prompt."""

import os

import yaml


class OpenAIPrompt:
    """OpenAI Prompt.

    This class is used to load and manage prompt templates for OpenAI.
    """

    def __init__(self, prompt_name: str) -> None:
        """Initialize the LangchainPrompt.

        :param prompt_name: The name of the prompt to load.
        """
        self._prompt_name = prompt_name
        self._prompt_template = self._get_template()

    def _get_template(self) -> list[dict[str, str]]:
        """"""
        prompt_file_path = os.path.join(
            os.path.dirname(__file__), "..", "prompt", f"{self._prompt_name}.yaml"
        )

        with open(prompt_file_path, "r", encoding="utf-8") as file:
            prompt = yaml.safe_load(file)

        template = prompt["template"]

        if isinstance(template, list) and all(
            isinstance(piece, str) and len(chat) == 2 for chat in template for piece in chat
        ):
            return [{chat[0]: chat[1]} for chat in template]

        raise ValueError("The provided template does not match the expected format.")

    def format(self, parameters: dict[str, str] | None) -> list[dict[str, str]]:
        """"""
        return [
            {"role": role, "content": template.format(**parameters)}
            for message in self._prompt_template
            for role, template in message.items()
        ]


if __name__ == "__main__":
    prompt_name = "summary"
    prompt = OpenAIPrompt(prompt_name=prompt_name)

    parameters = {"text": "안녕 들리니?"}
    messages = prompt.format(parameters=parameters)
    print(messages)
