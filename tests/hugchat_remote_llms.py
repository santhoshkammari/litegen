from src.ailite.main._model._base._hf_chatbot_custom import HFChatBot

from hugchat.login import Login

# Log in to huggingface and grant authorization to huggingchat
EMAIL = "backupsanthosh1@gmail.com"
PASSWD = "SK99@pass"
cookie_path_dir = "./cookies/" # NOTE: trailing slash (/) is required to avoid errors
sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

# Create your ChatBot
chatbot = HFChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

print([_.name for _ in chatbot.get_remote_llms()])