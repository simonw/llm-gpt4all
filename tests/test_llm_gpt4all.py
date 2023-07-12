from click.testing import CliRunner
from llm.cli import cli
import json


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
