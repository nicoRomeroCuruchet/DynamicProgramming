from setuptools import setup, find_packages

setup(
    name="dynamic-programming",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "cupy-cuda13x",
        "loguru",
        "matplotlib",
        "gymnasium[classic_control]",
    ],
)
