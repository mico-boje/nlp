from setuptools import setup, find_packages

setup(
    name='summarisation',
    version='0.1.0',
    description='',
    author='',
    author_email='mico.boeje@gmail.com',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'torch',
        'transformers',
        'pypdf',
        'tiktoken',
        'datasets',
        'tokenizers',
        'sentencepiece',
        'protobuf==3.20.0',
        'evaluate',
        'nltk',
        'rouge-score',
        'absl-py',
        'accelerate',
        'nvidia-ml-py3',
        # add additional libraries as needed
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu116'
    ]
)
