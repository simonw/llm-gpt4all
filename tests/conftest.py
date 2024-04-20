import json
import llm
import pytest

DUMMY_MODELS = [
    {
        "order": "a",
        "md5sum": "c87ad09e1e4c8f9c35a5fcef52b6f1c9",
        "name": "Llama 3 Instruct",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "filesize": "4661724384",
        "requires": "2.7.1",
        "ramrequired": "8",
        "parameters": "8 billion",
        "quant": "q4_0",
        "type": "LLaMA3",
        "description": '<ul><li>Fast responses</li><li>Chat based model</li><li>Accepts system prompts in Llama 3 format</li><li>Trained by Meta</li><li>License: <a href="https://llama.meta.com/llama3/license/">Meta Llama 3 Community License</a></li></ul>',
        "url": "https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "promptTemplate": "<|start_header_id|>user<|end_header_id|>\n\n%1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n%2<|eot_id|>",
        "systemPrompt": "",
    },
    {
        "order": "c",
        "md5sum": "97463be739b50525df56d33b26b00852",
        "name": "Mistral Instruct",
        "filename": "mistral-7b-instruct-v0.1.Q4_0.gguf",
        "filesize": "4108916384",
        "requires": "2.5.0",
        "ramrequired": "8",
        "parameters": "7 billion",
        "quant": "q4_0",
        "type": "Mistral",
        "systemPrompt": "",
        "description": "<strong>Strong overall fast instruction following model</strong><br><ul><li>Fast responses</li><li>Trained by Mistral AI<li>Uncensored</li><li>Licensed for commercial use</li></ul>",
        "url": "https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf",
        "promptTemplate": "[INST] %1 [/INST]",
    },
    {
        "order": "m",
        "md5sum": "0e769317b90ac30d6e09486d61fefa26",
        "name": "Mini Orca (Small)",
        "filename": "orca-mini-3b-gguf2-q4_0.gguf",
        "filesize": "1979946720",
        "requires": "2.5.0",
        "ramrequired": "4",
        "parameters": "3 billion",
        "quant": "q4_0",
        "type": "OpenLLaMa",
        "description": "<strong>Small version of new model with novel dataset</strong><br><ul><li>Instruction based<li>Explain tuned datasets<li>Orca Research Paper dataset construction approaches<li>Cannot be used commercially</ul>",
        "url": "https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf",
        "promptTemplate": "### User:\n%1\n\n### Response:\n",
        "systemPrompt": "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n",
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
