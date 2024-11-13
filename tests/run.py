messages = [{"role":"system","content":"you are doctor , your naem is kammari santhosh"},{"role":"user","content":"what is your name?"}]
from src.ailite import ai
print(ai(messages))