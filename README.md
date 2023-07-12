# llm-gpt4all

[![PyPI](https://img.shields.io/pypi/v/llm-gpt4all.svg)](https://pypi.org/project/llm-gpt4all/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-gpt4all?include_prereleases&label=changelog)](https://github.com/simonw/llm-gpt4all/releases)
[![Tests](https://github.com/simonw/llm-gpt4all/workflows/Test/badge.svg)](https://github.com/simonw/llm-gpt4all/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-gpt4all/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for the [GPT4All](https://gpt4all.io/) collection of models.

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-gpt4all
```
After installing the plugin you can see a new list of available models like this:

```bash
llm models list
```
The output will include something like this:
```
gpt4all: orca-mini-3b - Orca (Small), 1.80GB download, needs 4GB RAM (installed)
gpt4all: ggml-gpt4all-j-v1 - Groovy, 3.53GB download, needs 8GB RAM (installed)
gpt4all: nous-hermes-13b - Hermes, 7.58GB download, needs 16GB RAM (installed)
gpt4all: orca-mini-7b - Orca, 3.53GB download, needs 8GB RAM
gpt4all: ggml-model-gpt4all-falcon-q4_0 - GPT4All Falcon, 3.78GB download, needs 8GB RAM
gpt4all: ggml-vicuna-7b-1 - Vicuna, 3.92GB download, needs 8GB RAM
gpt4all: ggml-wizardLM-7B - Wizard, 3.92GB download, needs 8GB RAM
gpt4all: ggml-mpt-7b-base - MPT Base, 4.52GB download, needs 8GB RAM
gpt4all: ggml-mpt-7b-instruct - MPT Instruct, 4.52GB download, needs 8GB RAM
gpt4all: ggml-mpt-7b-chat - MPT Chat, 4.52GB download, needs 8GB RAM
gpt4all: ggml-replit-code-v1-3b - Replit, 4.84GB download, needs 4GB RAM
gpt4all: orca-mini-13b - Orca (Large), 6.82GB download, needs 16GB RAM
gpt4all: GPT4All-13B-snoozy - Snoozy, 7.58GB download, needs 16GB RAM
gpt4all: ggml-vicuna-13b-1 - Vicuna (large), 7.58GB download, needs 16GB RAM
gpt4all: ggml-nous-gpt4-vicuna-13b - Nous Vicuna, 7.58GB download, needs 16GB RAM
gpt4all: ggml-stable-vicuna-13B - Stable Vicuna, 7.58GB download, needs 16GB RAM
gpt4all: wizardLM-13B-Uncensored - Wizard Uncensored, 7.58GB download, needs 16GB RAM
```
Further details on these models can be found [in this Observable notebook](https://observablehq.com/@simonw/gpt4all-models).

## Usage

You can execute a model using the name displayed in the `llm models list` output. The model file will be downloaded the first time you attempt to run it.

```bash
llm -m orca-mini-7b '3 names for a pet cow'
```
The first time you run this you will see a progress bar:
```
 31%|█████████▋                        | 1.16G/3.79G [00:26<01:02, 42.0MiB/s]
```
On subsequent uses the model output will be displayed immediately.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

    cd llm-gpt4all
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
