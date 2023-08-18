from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "torch==2.0.1+cu118",
    "transformers==4.31.0",
    "tokenizers==0.13.3",
    "huggingface-hub==0.16.4"]

setup(
    name='zero_shot_classifier',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=INSTALL_REQUIRES,
)
