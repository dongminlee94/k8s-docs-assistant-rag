"""Prompt."""

import os

import yaml
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate


class LangchainPrompt:
    """Langchain Prompt.

    This class is used to load and manage prompt templates for Langchain.
    """

    def __init__(self, prompt_name: str) -> None:
        """Initialize the LangchainPrompt.

        :param prompt_name: The name of the prompt to load.
        """
        self._prompt_name = prompt_name
        self._prompt_template = self._get_template()

    def _get_template_type(self, template: str | list[list[str]] | list[tuple[str]]) -> str:
        """Determine the type of the template.

        :param template: The template to be checked.
        :returns: A string indicating the type of the template ('string' or 'chat').
        :raises ValueError: If the template does not match the expected formats.
        """
        if isinstance(template, str):
            return "string"
        elif isinstance(template, list) and all(
            isinstance(piece, str) and len(chat) == 2 for chat in template for piece in chat
        ):
            return "chat"

        raise ValueError("The provided template does not match the expected format for 'string' or 'chat'.")

    def _get_template(self) -> str | list[tuple[str]]:
        """Load the prompt template from a YAML file.

        :returns: The loaded template, either as a string or a list of tuples.
        """
        prompt_file_path = os.path.join(
            os.path.dirname(__file__), "..", "prompt", f"{self._prompt_name}.yaml"
        )

        with open(prompt_file_path, "r", encoding="utf-8") as file:
            prompt = yaml.safe_load(file)

        template = prompt["template"]

        if self._get_template_type(template=template) == "chat":
            template = [tuple(chat) for chat in template]

        return template

    @property
    def template_type(self) -> str:
        """Get the type of the loaded template.

        :returns: A string indicating the type of the template ('string' or 'chat').
        """
        return self._get_template_type(template=self._prompt_template)

    @property
    def template(self) -> BasePromptTemplate:
        """Get the prompt template in the appropriate Langchain format.

        :returns: The prompt template as a BasePromptTemplate object.
        """
        if self.template_type == "string":
            return PromptTemplate.from_template(self._prompt_template)
        elif self.template_type == "chat":
            return ChatPromptTemplate.from_messages(self._prompt_template)


if __name__ == "__main__":
    prompt_name = "summarizer"
    prompt = LangchainPrompt(prompt_name=prompt_name)
    print(prompt.template_type)
    print(prompt.template)
