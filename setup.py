import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="typegpt",
    version="0.2.1",
    author="Alexander Eichhorn",
    author_email="",
    description="TypeGPT - Make GPT safe for production",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexeichhorn/typegpt",
    install_requires=[
        "typing_extensions>=4.1.0",
        "inflect>=7.0.0",
        "tiktoken>=0.5.0",
        "openai>=1.1.0",
    ],
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"],
    python_requires=">=3.10",
)
