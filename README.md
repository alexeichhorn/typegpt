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


### Output Types

Your output type can contain string, integer, float, boolean, or lists of these. It is also possible to mark elements as optional. You can also provide default values.

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
Here, the parser will parse `description` if the LLM returns it, but won't require it. It is `None` by default. The same holds for `title`, as we have a default value here.


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

By default, the library always expects only one line response per element. You can overwrite this by setting `multiline=True` in `LLMOutput`:
```python
class Output(BaseLLMResponse):
    description: str  = LLMOutput(instruction="A description for the recipe.", multiline=True)
    items: list[str] = LLMArrayOutput(expected_count=5, instruction=lambda pos: f"The {pos.ordinal} item in the list", multiline=True)

```





## Advanced Usage

### Automatic Prompt Reduction
You might have a prompt that uses unpredictable many tokens due to some potentially large dependencies. To make sure your prompt always fits wihtin the LLMs token limit, you can implement the function `reduce_if_possible` inside your prompt class:
```python
class Summaryrompt(PromptTemplate):

    def __init__(self, article: str):
        self.article = article

    def system_prompt(self) -> str:
        return "Summarize the given news article"

    def user_prompt(self) -> str:
        return f"ARTICLE: {self.article}"

    def reduce_if_possible(self) -> bool:
        if len(self.article) > 100:
            # remove last 100 characters at a time
            self.article = self.article[:100]
            return True
        return False

    class Output(BaseLLMResponse):
        summary: str
```
Inside the `reduce_if_possible` function you should reduce the size of your prompt in small steps and return `True` if successfully reduced. The function gets called over and over again until the prompt fits.
When calling the openai `generate_output` function this automatically ensures the prompt fits for the given models. Additionally, you can also specify a custom input token limit with the same effect to save costs: `client.chat.completions.generate_output(..., max_input_tokens=2000)`.


### Automatic Retrying

In some cases GPT might still return an output not following the schema correctly. When this happens the OpenAI client throws `LLMParseException`. To automatically retry whenever the output doesn't meet your schema, you can set `retry_on_parse_error` to the number of retries you want to allow:
```python
out = client.chat.completions.generate_output("gpt-3.5-turbo", prompt=prompt, ..., retry_on_parse_error=3)
```
Now the library retries calling GPT 3 times before throwing the error. However, make sure you only use this whenever the temperature is not zero.




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

This library automatically generates a LLM-compatible schema from your defined output class and adds instructions to the end of the system prompt to follow this schema.
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
See how the plural "mice" gets automatically converted into singular "mouse" to not confuse the language model.




## Coming Soon

- Support for output classes within output classes, especially arrays
