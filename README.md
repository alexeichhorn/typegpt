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

If you are using OpenAI as your LLM provider, simply add a "Condom" to your client class name (e.g. `OpenAICondom`, `AsyncOpenAICondom`, or `AzureOpenAICondom`) to make it safe. You can still use it as you would have used it before, but can now also call the `generate_output` function for chat completions like this to generate the output object:
```python
from gpt_condom.openai import OpenAICondom

prompt = ExamplePrompt("The young athlete demonstrated exceptional skill and agility on the field.")

client = OpenAICondom(api_key="<your api key>") # subclass of `OpenAI`

output = client.chat.completions.generate_output(model="gpt-3.5-turbo", prompt=prompt, max_output_tokens=1000)
```

And you get a nice output like this:
```python
Output(adjectives=['young', 'exceptional'], nouns=['athlete', 'skill', 'agility', 'field'], verbs=['demonstrated'])
```


## Advanced Usage

### Full Static Typesafety

```python
prompt = ExamplePrompt(...)
output = client.chat.completions.generate_output(model="gpt-4", prompt=prompt, ...)
```
Due to Python's limited type system, the output type is of type `BaseLLMResponse` instead of the explicit subclass `ExamplePrompt.Output`. To achieve full type-safety in your code, simply add the parameter `output_type=ExamplePrompt.Output`:
```python
prompt = ExamplePrompt(...)
output = client.chat.completions.generate_output(model="gpt-4", prompt=prompt, output_type=ExamplePrompt.Output, ...)
```
This parameter isn't simply a type decorator. It can also be used to overwrite the actual output type, GPT tries to predict.


### Azure

Make sure to use the `AzureChatModel` as model when generating the output, which consists of the deployment_id and the corresponding base model (this is used for automatically reducing prompts if needed).
```python
from gpt_condom.openai import AzureChatModel, AzureOpenAICondom

client = client = AzureOpenAICondom(
    azure_endpoint="<your azure endpoint>",
    api_key="<your api key>",
    api_version="2023-05-15",
)

out = client.chat.completions.generate_output(model=AzureChatModel(deployment_id="gpt-35-turbo", base_model="gpt-3.5-turbo"), prompt=prompt, max_output_tokens=1000)
```

### Non-OpenAI LLM support

Any LLM that has a notion of system and user prompt can use this library. Simply generate the system and user messages (including the schema prompt) like this:
```python
messages = prompt.generate_messages(
    token_limit=max_prompt_length, token_counter=lambda messages: num_tokens_from_messages(messages)
)
```
where `max_prompt_length` is the maximum amount of tokens the prompt is allowed to use and `num_tokens_from_messages` needs to be a function that counts the predicted token usage for a given list of messages. Simply return `0` here, if you don't want to automatically reduce the size of a prompt.

Use the generated messages to call your LLM. Use the completion string you received back like this to parse it into the desired output class:
```python
out = ExamplePrompt.Output.parse_response(completion)
```





## How it works

...


## Coming Soon

- Support for output classes within output classes, especially arrays
