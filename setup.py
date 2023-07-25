from setuptools import setup
import os

VERSION = "0.1.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="llm-gpt4all",
    description="Plugin for LLM adding support for GPT4ALL models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Simon Willison",
    url="https://github.com/simonw/llm-gpt4all",
    project_urls={
        "Issues": "https://github.com/simonw/llm-gpt4all/issues",
        "CI": "https://github.com/simonw/llm-gpt4all/actions",
        "Changelog": "https://github.com/simonw/llm-gpt4all/releases",
    },
    license="Apache License, Version 2.0",
    classifiers=["License :: OSI Approved :: Apache Software License"],
    version=VERSION,
    modules=["llm_gpt4all"],
    entry_points={"llm": ["gpt4all = llm_gpt4all"]},
    install_requires=[
        "llm>=0.5",
        "gpt4all",
        "httpx",
    ],
    extras_require={"test": ["pytest"]},
    python_requires=">=3.9",
)
