from concurrent.futures.thread import ThreadPoolExecutor

from langchain_ollama import ChatOllama
from wordllama import WordLlama
from visionlite import vision


class AIValidator:
    def __init__(self,wllm=None,llm=None):
        self.wllm = wllm or WordLlama.load()
        self.llm = llm or ChatOllama(model="llama3.2:latest")

    def get_content_splits(self,content:str):
        return self.wllm.split(content) if content else []

    def generate_google_question(self,texts):
        _sys_prompt = ("You are Content queries generator,"
                       "based on input text generate a query to validate in realtime"
                       "start providing single query directly")
        _messages = [[{"role":"system","content":_sys_prompt},
                     {"role":"user","content":_user_prompt}] for _user_prompt in texts]
        return [_.content for _ in self.llm.batch(_messages)]

    @classmethod
    def search_google(cls,queries:list):
        """Threadpool executor search using vision"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(vision, queries))

    def validate_queries(self,queries:list,results:list):
        """Validate queries using LLM"""
        _sys_prompt = ("You are query Validator reasoner expert.,"
                       "based on input text Validate the query based on the realtime data"
                       "start providing single reason directly")
        _messages = [[{"role": "system", "content": _sys_prompt},
                      {"role": "user", "content": f"Googleresults (realtime data): \n\n {r}"
                                                  f"query: {q}"}] for q,r in zip(queries,results)]
        return [_.content for _ in self.llm.batch(_messages)]

    def get_validation_scores(self,questions:list,validations:list):
        _sys_prompt = ("You are Content Critic scorer,"
                       "based the reason provided a score of [0,0.3,0.5,0.8,1]"
                       "start providing score directly as float number with reason why you gave this score")
        _messages = [[{"role": "system", "content": _sys_prompt},
                      {"role": "user", "content": "reason from realtime human:\n"
                                                  f"{v}\n\n"
                                                  f"query : {q}"}] for q,v in zip(questions,validations)]
        return [_.content for _ in self.llm.batch(_messages)]

    def update_chunks(self,queries:list,validations:list,scores:list):
        _sys_prompt = ("You are queries updater "
                       "based the reason provided reflect yourself and udpate the query"
                       "start by providing the updated query directly")
        _messages = [[{"role": "system", "content": _sys_prompt},
                      {"role": "user", "content": "reason from realtime human:\n"
                                                  f"{v}\n\n"
                                                  f"with score of {s}\n\n"
                                                  f"query : {q}"}] for q, v, s in zip(queries, validations,scores)]
        return [_.content for _ in self.llm.batch(_messages)]

    def get_updated_content(self,content:str,return_list:bool=False):
        chunks = validator.get_content_splits(content)
        google_questions = validator.generate_google_question(chunks)
        google_results = validator.search_google(google_questions)
        validations = validator.validate_queries(google_questions, google_results)
        validation_scores = validator.get_validation_scores(chunks, validations)
        updated_chunks = validator.update_chunks(chunks, validations, validation_scores)
        if return_list:
            return updated_chunks
        return "".join(updated_chunks)

__client = AIValidator()
ai_validate = __client.get_updated_content

if __name__ == '__main__':
    llm = ChatOllama(base_url="http://192.168.170.76:11434",model="qwen2.5:7b-instruct")
    validator = AIValidator(llm=llm)
    print(validator.get_updated_content("Bananas are not classified as berries, while strawberries are true berries."))




