from gpt4all import GPT4All as _GPT4All
from pathlib import Path
from typing import Optional
import httpx
import urllib3
import json
import llm
import os
import sys
import time
from typing import List, Optional, Tuple


class GPT4All(_GPT4All):
    # Switch verbose default to False
    @staticmethod
    def retrieve_model(
        model_name: str,
        model_path: Optional[str] = None,
        allow_download: bool = True,
        verbose: bool = False,
    ) -> str:
        return _GPT4All.retrieve_model(model_name, model_path, allow_download, verbose)


def get_gpt4all_models():
    return fetch_cached_json(
        url="https://gpt4all.io/models/models.json",
        path=llm.user_dir() / "gpt4all_models.json",
        cache_timeout=3600,
    )


@llm.hookimpl
def register_models(register):
    raw_models = get_gpt4all_models()
    # Sort them by if they are installed or not
    models = [Gpt4AllModel(model) for model in raw_models]
    models.sort(
        key=lambda model: (
            not model.is_installed(),
            model.filesize_bytes(),
            model.model_id,
        )
    )
    for model in models:
        register(model)


class Gpt4AllModel(llm.Model):
    can_stream = True

    def __init__(self, details):
        self._details = details
        self.model_id = details["filename"].split(".")[0]

    def prompt_template(self):
        return (
            self._details.get("promptTemplate") or "### Human: \n%1\n### Assistant:\n"
        )

    def system_prompt(self):
        return self._details.get("systemPrompt")

    def build_prompt_blocks_and_system(
        self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]
    ) -> Tuple[List[str], str]:
        blocks = []

        # Simplified handling of system prompts: use the one from prompt.system, or the
        # one from the first message in the conversation, or the default for the model.
        # Ignore the case where the system prompt changed mid-conversation.
        system_prompt = None
        if prompt.system:
            system_prompt = prompt.system

        if conversation is not None:
            for response in conversation.responses:
                if response.prompt.system:
                    system_prompt = response.prompt.system
                    break

        if system_prompt is None:
            system_prompt = self.system_prompt()

        template = self.prompt_template()
        # Special case to add <|im_end|> if it looks necessary
        template_end = ""
        if "<|im_start|>" in template and template.count(
            "<|im_start|>"
        ) - 1 == template.count("<|im_end|>"):
            template_end = "<|im_end|>"

        if conversation is not None:
            for prev_response in conversation.responses:
                blocks.append(template.replace("%1", prev_response.prompt.prompt))
                blocks.append(prev_response.text() + template_end)

        # Add the user's prompt
        blocks.append(template.replace("%1", prompt.prompt))

        return blocks, system_prompt

    def execute(self, prompt, stream, response, conversation):
        with SuppressOutput():
            blocks, system = self.build_prompt_blocks_and_system(prompt, conversation)
            text_prompt = "".join(blocks)
            if system:
                text_prompt = f"{system}\n{text_prompt}"
            response.response_json = {"full_prompt": text_prompt}
            gpt_model = GPT4All(self.filename())
            output = gpt_model.generate(text_prompt, max_tokens=400, streaming=True)
            yield from output

    def filename(self):
        return self._details["filename"]

    def filesize_bytes(self):
        return int(self._details["filesize"])

    def is_installed(self):
        try:
            GPT4All.retrieve_model(
                self._details["filename"], allow_download=False, verbose=False
            )
            return True
        except ValueError:
            return False

    def __str__(self):
        installed = " (installed)" if self.is_installed() else ""
        return "gpt4all: {} - {}, {} download, needs {}GB RAM{}".format(
            self.model_id,
            self._details["name"],
            human_readable_size(self.filesize_bytes()),
            self._details["ramrequired"],
            installed,
        )


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)

    # Create directories if not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        # Get the file's modification time
        mod_time = path.stat().st_mtime
        # Check if it's more than the cache_timeout old
        if time.time() - mod_time < cache_timeout:
            # If not, load the file
            with open(path, "r") as file:
                return json.load(file)

    # Try to download the data
    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()  # This will raise an HTTPError if the request fails

        # If successful, write to the file
        with open(path, "w") as file:
            json.dump(response.json(), file)

        return response.json()
    except (httpx.HTTPError, urllib3.exceptions.NameResolutionError):
        # If there's an existing file, load it
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            # If not, raise an error
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


def human_readable_size(size_bytes):
    if size_bytes == 0:
        return "0B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0

    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1

    return "{:.2f}{}".format(size_bytes, size_name[i])


class SuppressOutput:
    def __enter__(self):
        # Save a copy of the current file descriptors for stdout and stderr
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        # Open a file to /dev/null
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Replace stdout and stderr with /dev/null
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)

        # Writes to sys.stdout and sys.stderr should still work
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = os.fdopen(self.stdout_fd, "w")
        sys.stderr = os.fdopen(self.stderr_fd, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout and stderr to their original state
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)

        # Close the saved copies of the original stdout and stderr file descriptors
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)

        # Close the file descriptor for /dev/null
        os.close(self.devnull_fd)

        # Restore sys.stdout and sys.stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
