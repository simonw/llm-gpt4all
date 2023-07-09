from click.testing import CliRunner
from llm.cli import cli
import json


def test_plugin_is_installed():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins"])
    assert result.exit_code == 0, result.output
    names = [plugin["name"] for plugin in json.loads(result.output)]
    assert "llm-gpt4all" in names
