from litegen import completion,pp_completion
import os

os.environ['OPENAI_API_KEY'] = 'huggingchat'

pp_completion(model="NousResearch/Hermes-3-Llama-3.1-8B",
            messages="what is 2+10? explain.",
            system_prompt='you are helpfull assistan')

