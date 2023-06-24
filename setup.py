import pathlib

from setuptools import setup

import autodocs

LOCAL = pathlib.Path(__file__).parent
README = ""  # (LOCAL / "README.md").read_text()

try:
    requirements = open("requirements.txt", "r").readlines()
except FileNotFoundError:
    requirements = []

try:
    dev_requirements = open("dev_requirements.txt", "r").readlines()
except FileNotFoundError:
    dev_requirements = []


setup(
    name=autodocs.__name__,
    version=autodocs.__version__,
    description=autodocs.__description__,
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["autodocs"],
    requirements=requirements,
    extras_require={"dev": dev_requirements},
)
