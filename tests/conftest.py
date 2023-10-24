import json
import llm
import pytest

DUMMY_MODELS = [
    {
        "name": "GPT4All Falcon",
        "filename": "ggml-model-gpt4all-falcon-q4_0.bin",
        "filesize": "4061641216",
        "ramrequired": "8",
        "promptTemplate": "### Instruction:\n%1\n### Response:\n",
        "systemPrompt": None,
    },
    {
        "name": "MPT Chat",
        "filename": "ggml-mpt-7b-chat.bin",
        "filesize": "4854401050",
        "ramrequired": "8",
        "promptTemplate": "<|im_start|>user\n%1<|im_end|><|im_start|>assistant\n",
        "systemPrompt": "<|im_start|>system\n- You are a helpful assistant chatbot trained by MosaicML.\n- You answer questions.\n- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>",
    },
    {
        "name": "Orca",
        "filename": "orca-mini-7b.ggmlv3.q4_0.bin",
        "filesize": "3791749248",
        "ramrequired": "8",
        "promptTemplate": "### User:\n%1\n### Response:\n",
        "systemPrompt": "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n",
    },
    {
        "name": "Groovy",
        "filename": "ggml-gpt4all-j-v1.3-groovy.bin",
        "filesize": "3785248281",
        "ramrequired": "8",
        "promptTemplate": None,
        "systemPrompt": None,
    },
]


@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    # Write out the models.json file
    (llm.user_dir() / "gpt4all_models2.json").write_text(
        json.dumps(DUMMY_MODELS), "utf-8"
    )
