from click.testing import CliRunner
from llm.cli import cli
import llm
import json
import pytest


def test_plugin_is_installed():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins"])
    assert result.exit_code == 0, result.output
    names = [plugin["name"] for plugin in json.loads(result.output)]
    assert "llm-gpt4all" in names


def test_llm_models():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    for fragment in (
        "gpt4all: Meta-Llama-3-8B-Instruct - Llama 3 Instruct, 4.34GB download, needs 8GB RAM",
        "gpt4all: mistral-7b-instruct-v0 - Mistral Instruct, 3.83GB download, needs 8GB RAM",
    ):
        assert fragment in result.output


@pytest.mark.parametrize(
    "model_id,expected_blocks",
    (
        (
            "Meta-Llama-3-8B-Instruct",
            [
                "<|start_header_id|>user<|end_header_id|>\n\ninput 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n%2<|eot_id|>",
                "response 1",
                "<|start_header_id|>user<|end_header_id|>\n\ninput 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n%2<|eot_id|>",
            ],
        ),
        (
            "mistral-7b-instruct-v0",
            ["[INST] input 1 [/INST]", "response 1", "[INST] input 2 [/INST]"],
        ),
        (
            "orca-mini-3b-gguf2-q4_0",
            [
                "### User:\ninput 1\n\n### Response:\n",
                "response 1",
                "### User:\ninput 2\n\n### Response:\n",
            ],
        ),
    ),
)
def test_conversation_prompt_blocks(model_id, expected_blocks):
    model = llm.get_model(model_id)
    conversation = model.conversation()
    conversation.responses = [
        llm.Response.fake(model, "input 1", None, "response 1"),
    ]
    blocks, system = model.build_prompt_blocks_and_system(
        llm.Prompt("input 2", model), conversation
    )
    assert blocks == expected_blocks
    assert system == model.system_prompt()
