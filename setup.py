from pathlib import Path

from setuptools import setup


setup(
    name="msce",
    version="0.1.0",
    description="Memory-Skill Co-Evolution for long-horizon LLM agents",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["msce"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "requests>=2.31",
    ],
)
