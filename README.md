# GPT Condom - Make GPT safe for production

It's inheritly hard to make LLMs output in a consistent structure. GPT Condom makes this as easy as defining a class in Python. 


## Installation

```bash
pip install gpt-condom
```

## Usage

Define your prompt and desired output schema as a subclass in Python:

```python
from gpt_condom import BaseLLMResponse, PromptTemplate

class ExamplePrompt(PromptTemplate):

    def __init__(self, sentence: str):
        self.sentence = sentence

    def system_prompt(self) -> str:
        return "Given a sentence, extract sentence parts."

    def user_prompt(self) -> str:
        return self.sentence

    class Output(BaseLLMResponse):
        adjectives: list[str]
        nouns: list[str]
        verbs: list[str]
```

If you are using OpenAI as your LLM provider, you can execute the prompt as easy as this:
```python
from gpt_condom.openai import OpenAICondom

prompt = ExamplePrompt("The young athlete demonstrated exceptional skill and agility on the field.")

client = OpenAICondom(api_key="<your api key>") # subclass of `OpenAI`

output = client.chat.completions.generate_output("gpt-3.5-turbo", prompt=prompt, max_output_tokens=1000)
```

And you get a nice output like this:
```python
Output(adjectives=['young', 'exceptional'], nouns=['athlete', 'skill', 'agility', 'field'], verbs=['demonstrated'])
```


