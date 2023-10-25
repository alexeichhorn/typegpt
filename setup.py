import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="gpt-condom",
    version="0.0.1",
    author="Alexander Eichhorn",
    author_email="",
    description="GPT Condom - Make GPT safe for production",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexeichhorn/gpt-condom",
    install_requires=[
        "typing_extensions>=4.1.0",
        "inflect>=7.0.0",
        "tiktoken>=0.5.0",
        "openai>=0.28.0",
    ],
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "License :: MIT License", "Operating System :: OS Independent"],
    python_requires=">=3.10",
)
