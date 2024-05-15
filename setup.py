from setuptools import setup, find_packages

setup(
    name='PolicyIteration',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'tqdm',
        'gymnasium',
        'scipy',
        'loguru',
        'matplotlib',
        'gymnasium[classic_control]',
    ],
)

