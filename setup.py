from setuptools import setup, find_packages

setup(
    name="classic_control",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy>=1.21.0",
        "tqdm",
        "scipy",
        "loguru",
        "matplotlib",
        "gymnasium[classic_control]"
    ],
)