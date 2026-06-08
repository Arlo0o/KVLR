from typing import List
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def fetch_requirements(paths) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    if not isinstance(paths, list):
        paths = [paths]
    requirements = []
    for path in paths:
        with open(ROOT / path, "r") as fd:
            requirements += [r.strip() for r in fd.readlines()]
    return requirements


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open(ROOT / "README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="kvlr-surgical-video",
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "assets",
            "configs",
            "docs",
            "eval",
            "evaluation_results",
            "gradio",
            "logs",
            "notebooks",
            "outputs",
            "pretrained_models",
            "samples",
            "scripts",
            "*.egg-info",
        )
    ),
    description="Kinematic-to-Visual Action Routing for surgical video generation",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="",
    project_urls={},
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
