from ailite.model import completion
from ailite.model._oai import OmniLLMClient

client = OmniLLMClient(gpu=True)
completion = client.completion


print(completion(model='qwen2.5:0.5b-instruct',
                 messages='hi, what is weather today in gurgaon',
                 ))
