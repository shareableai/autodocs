import json
from typing import List

from langchain.prompts import StringPromptTemplate
from pydantic import validator, BaseModel

from autodocs.prompts.tagging.examples import TagGenerationExampleSelector


class TagGenerationPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in source code and function calls as input, and identifies the relevant
    tags for the model"""

    _prompt = prompt_template = """
You are a Machine Learning Engineer describing model code.

# Rules
1. Interpret [1] to mean the following function had a stack depth of 1.
2. For each tag, write a number from 0.1 to 1.0 indicating how accurately the tag describes the code provided.
3. If you cannot tell, write 0.

{all_tags}

===
{examples}

## Functions Executed
{trace}

## Function Definitions
{functions}

## Tags
"""

    _example_prompt = """
## Functions Executed
{trace}

## Function Definitions
{functions}

## Tags
{example_tags}
===
"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 2 or "trace" not in v or "functions" not in v:
            raise ValueError("trace and functions must be provided")
        return v

    @staticmethod
    def prompt_tags() -> List[str]:
        with open("/tagging/data/config.json") as f:
            return json.load(f)["tags"]

    @classmethod
    def _format_example_tags(cls, example_tags: str) -> str:
        identified_tags = {
            tag.split(": ")[0]: tag.split(": ")[1] for tag in example_tags.split(", ")
        }
        all_tags = {tag: 0 for tag in cls.prompt_tags()}
        return "\n".join(
            [
                f"{idx}. {tag}: {tag_value}"
                for (idx, (tag, tag_value)) in enumerate(
                    (all_tags | identified_tags).items()
                )
            ]
        )

    def format(self, trace: str, functions: str) -> str:
        examples = TagGenerationExampleSelector.examples(trace, functions)
        formatted_examples = [
            self._example_prompt.format(
                trace=example["trace"],
                functions=example["functions"],
                example_tags=self._format_example_tags(example["tags"]),
            )
            for example in examples
        ]
        prompt = self._prompt.format(
            all_tags=[f"{idx}. {tag}" for (idx, tag) in enumerate(self.prompt_tags())],
            trace=trace,
            functions=functions,
            examples="".join(formatted_examples),
        )
        return prompt

    def _prompt_type(self) -> str:
        return "tag-generator"
