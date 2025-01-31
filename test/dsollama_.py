import random

from litegen import LLM
import os

# os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY']  = "huggingchat"

genai = LLM()

query = "train a ml model using sklearn, i want step by step , make it neat, do it step by step each time do one step only, now do one step"
res = ""
code = ""
max_turns = 10
turn =0
while res=="" or 'IMPROVEMENT' in res:
    turn+=1
    if turn>max_turns:
        print(res)
        print(code)
        break

    start = f"write python code to {query}, "
    middle = f'Improvements suggested: {res} and previous code is {code}' if res else ""
    end = f"start only python code no explanation  , staty by ```python"
    code += genai(model="Qwen/Qwen2.5-Coder-32B-Instruct",
                    prompt  = start+middle+end
                 )

    res += genai(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        prompt=f"""i will provide you code and user quesiton,
    write down IMPROVEMENTS and points if any changes required to make the code better or else say DONE 
    query was {query}
    
    code i have is 
    {code}
    
    i don't want any code i just want the points to know thats it.
    write down IMPROVEMENTS and points if any changes required to make the code better or else say DONE
    make sure query is satisfied and suggest improvement for clearness
    if steps are not completed and steps are exists then show that steps as IMPROVEMENTS 
    """)

