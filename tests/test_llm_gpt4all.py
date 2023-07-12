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
        "gpt4all: ggml-gpt4all-j-v1 - Groovy, 3.53GB download, needs 8GB RAM",
        "gpt4all: orca-mini-7b - Orca, 3.53GB download, needs 8GB RAM",
        "gpt4all: ggml-model-gpt4all-falcon-q4_0 - GPT4All Falcon, 3.78GB download, needs 8GB RAM",
        "gpt4all: ggml-mpt-7b-chat - MPT Chat, 4.52GB download, needs 8GB RAM",
    ):
        assert fragment in result.output


@pytest.mark.parametrize(
    "model_id,expected_blocks",
    (
        (
            "ggml-gpt4all-j-v1",
            # This has no promptTemplate, so default display
            [
                "### Human: \ninput 1\n### Assistant:\n",
                "response 1",
                "### Human: \ninput 2\n### Assistant:\n",
            ],
        ),
        (
            "orca-mini-7b",
            # This has no promptTemplate, so default display
            [
                "### User:\ninput 1\n### Response:\n",
                "response 1",
                "### User:\ninput 2\n### Response:\n",
            ],
        ),
        (
            "ggml-mpt-7b-chat",
            # This has no promptTemplate, so default display
            [
                "<|im_start|>user\ninput 1<|im_end|><|im_start|>assistant\n",
                "response 1<|im_end|>",
                "<|im_start|>user\ninput 2<|im_end|><|im_start|>assistant\n",
            ],
        ),
        (
            "ggml-model-gpt4all-falcon-q4_0",
            # This has no promptTemplate, so default display
            [
                "### Instruction:\ninput 1\n### Response:\n",
                "response 1",
                "### Instruction:\ninput 2\n### Response:\n",
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
