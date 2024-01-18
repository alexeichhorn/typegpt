# TypeGPT - Making GPT Safe for Production

It is inherently difficult to produce outputs from LLMs in a consistent structure. TypeGPT simplifies this process to be as easy as defining a class in Python.

Powering our own projects, such as [BoostSEO](https://boostseo.ai)


## Installation

```bash
pip install typegpt
```

## Usage

Define your prompt and desired output schema as a subclass in Python:

```python
from typegpt import BaseLLMResponse, PromptTemplate

class ExamplePrompt(PromptTemplate):

    def __init__(self, sentence: str):
        self.sentence = sentence

    def system_prompt(self) -> str:
        return "Given a sentence, extract sentence parts."

    def user_prompt(self) -> str:
        return self.sentence

    class Output(BaseLLMResponse):
        num_sentences: int
        adjectives: list[str]
        nouns: list[str]
        verbs: list[str]
```

If you are using OpenAI as your LLM provider, simply replace the OpenAI client class name with the subclass `TypeOpenAI` (for async use `AsyncTypeOpenAI`, or for Azure use `TypeAzureOpenAI`/`AsyncTypeAzureOpenAI`) to make it safe. You can still use it as you would have before, but you can now also call the `generate_output` function for chat completions like this to generate the output object:

```python
from typegpt.openai import TypeOpenAI

prompt = ExamplePrompt("The young athlete demonstrated exceptional skill and agility on the field.")

client = TypeOpenAI(api_key="<your api key>") # subclass of `OpenAI`

output = client.chat.completions.generate_output(model="gpt-3.5-turbo", prompt=prompt, max_output_tokens=1000)
```

And you get a nice output like this:
```python
Output(num_sentences=1, adjectives=['young', 'exceptional'], nouns=['athlete', 'skill', 'agility', 'field'], verbs=['demonstrated'])
```


### Output Types

Your output type can contain string, integer, float, boolean, or lists of these. It is also possible to mark elements as optional. Default values can be provided as well.

#### Example 1
```python
class Output(BaseLLMResponse):
    title: str = "My Recipe"
    description: str | None
    num_ingredients: int
    ingredients: list[int]
    estimated_time: float
    is_oven_required: bool
```
Here, the parser will parse `description` if the LLM returns it, but won't require it. It is `None` by default. The same holds for `title`, as it has a default value.


#### Example 2

You can also define more restrictions or give the LLM more information for some elements:

```python
class Output(BaseLLMResponse):
    title: str = LLMOutput(instruction="The title for the recipe.")
    description: str | None = LLMOutput(instruction="An optional description for the recipe.")
    num_ingredients: int
    ingredients: list[int] = LLMArrayOutput(expected_count=(1, 5), instruction=lambda pos: f"The id of the {pos.ordinal} ingredient") # between 1 and 5 ingredients expected (and required at parse time)
    estimated_time: float = LLMOutput(instruction="The estimated time to cook")
    is_oven_required: bool
```

### Example 3

By default, the library always expects only one line response per element. You can override this by setting `multiline=True` in `LLMOutput`:
```python
class Output(BaseLLMResponse):
    description: str  = LLMOutput(instruction="A description for the recipe.", multiline=True)
    items: list[str] = LLMArrayOutput(expected_count=5, instruction=lambda pos: f"The {pos.ordinal} item in the list", multiline=True)

```

### Example 4

You can nest response types. Note that you need to use `BaseLLMArrayElement` for classes that you want to nest inside a list. To add instructions inside an element of `BaseLLMArrayElement`, you must use `LLMArrayElementOutput` instead of `LLMOutput`.

```python
class Output(BaseLLMResponse):

    class Item(BaseLLMArrayElement):

        class Description(BaseLLMResponse):
            short: str | None
            long: str

        title: str
        description: Description
        price: float = LLMArrayElementOutput(instruction=lambda pos: f"The price of the {pos.ordinal} item")

    items: list[Item]
    count: int
```





## Advanced Usage

### Automatic Prompt Reduction
You might have a prompt that uses an unpredictably large number of tokens due to potentially large dependencies. To ensure your prompt always fits within the LLM's token limit, you can implement the function `reduce_if_possible` inside your prompt class:
```python
class SummaryPrompt(PromptTemplate):

    def __init__(self, article: str):
        self.article = article

    def system_prompt(self) -> str:
        return "Summarize the given news article"

    def user_prompt(self) -> str:
        return f"ARTICLE: {self.article}"

    def reduce_if_possible(self) -> bool:
        if len(self.article) > 100:
            # remove last 100 characters at a time
            self.article = self.article[:-100]
            return True
        return False

    class Output(BaseLLMResponse):
        summary: str
```

Inside the `reduce_if_possible` function, you should reduce the size of your prompt in small steps and return `True` if successfully reduced. The function is called repeatedly until the prompt fits. When calling the OpenAI `generate_output` function, this automatically ensures the prompt is suitable for the given models. Additionally, you can specify a custom input token limit with the same effect to save costs: `client.chat.completions.generate_output(..., max_input_tokens=2000)`.


### Automatic Retrying

In some cases, GPT might still return an output that does not follow the schema correctly. When this occurs, the OpenAI client throws an `LLMParseException`. To automatically retry when the output does not meet your schema, you can set `retry_on_parse_error` to the number of retries you want to allow:
```python
out = client.chat.completions.generate_output("gpt-3.5-turbo", prompt=prompt, ..., retry_on_parse_error=3)
```
Now, the library will attempt to call GPT three times before throwing an error. However, ensure you only use this when the temperature is not zero.




### Full Static Type Safety

```python
prompt = ExamplePrompt(...)
output = client.chat.completions.generate_output(model="gpt-4", prompt=prompt, ...)
```
Due to Python's limited type system, the output type is of type `BaseLLMResponse` instead of the explicit subclass `ExamplePrompt.Output`. To achieve full type safety in your code, simply add the parameter `output_type=ExamplePrompt.Output`:
```python
prompt = ExamplePrompt(...)
output = client.chat.completions.generate_output(model="gpt-4", prompt=prompt, output_type=ExamplePrompt.Output, ...)
```
This parameter is not merely a type decorator. It can also be used to overwrite the actual output type that GPT attempts to predict.


### Azure

e sure to use the `AzureChatModel` as the model when generating the output, which consists of the deployment_id and the corresponding base model (this is used for automatically reducing prompts if needed).
```python
from typegpt.openai import AzureChatModel, TypeAzureOpenAI

client = TypeAzureOpenAI(
    azure_endpoint="<your azure endpoint>",
    api_key="<your api key>",
    api_version="2023-05-15",
)

out = client.chat.completions.generate_output(model=AzureChatModel(deployment_id="gpt-35-turbo", base_model="gpt-3.5-turbo"), prompt=prompt, max_output_tokens=1000)
```

### Non-OpenAI LLM support

Any LLM that has a notion of system and user prompts can use this library. Generate the system and user messages (including the schema prompt) like this:
```python
messages = prompt.generate_messages(
    token_limit=max_prompt_length, token_counter=lambda messages: num_tokens_from_messages(messages)
)
```
where `max_prompt_length` is the maximum number of tokens the prompt is allowed to use, and `num_tokens_from_messages` needs to be a function that counts the predicted token usage for a given list of messages. Return `0` here if you do not want to automatically reduce the size of a prompt.

Use the generated messages to call your LLM. Parse the completion string you receive back into the desired output class like this:
```python
out = ExamplePrompt.Output.parse_response(completion)
```





## How it works

This library automatically generates an LLM-compatible schema from your defined output class and adds instructions to the end of the system prompt to adhere to this schema.
For example, for the following abstract prompt:
```python
class DemoPrompt(PromptTemplate):

    def system_prompt(self) -> str:
        return "This is a system prompt"

    def user_prompt(self) -> str:
        return "This is a user prompt"

    class Output(BaseLLMResponse):
        title: str
        description: str = LLMOutput("Custom instruction")
        mice: list[str]
```

The following system prompt will be generated:
```
This is a system prompt

Always return the answer in the following format:
"""
TITLE: <Put the title here>
DESCRIPTION: <Custom instruction>
MOUSE 1: <Put the first mouse here>
MOUSE 2: <Put the second mouse here>
...
"""
```
Notice how the plural "mice" is automatically converted to the singular "mouse" to avoid confusing the language model.

