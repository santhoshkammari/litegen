from src.ailite import ai

for chunk in ai(model="Qwen/Qwen2.5-Coder-32B-Instruct",
         prompt_or_messages='python code to sum two numpy arrays ',stream=True):
    print(chunk,end= "",flush=True)