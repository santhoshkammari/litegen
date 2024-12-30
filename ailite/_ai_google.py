from visionlite import visionai

def ai_google(query,llm=None):
    res = visionai(query,llm=llm
               )
    return llm.invoke(
    f"based on result: {res} , answer the query clearly , query: {query}").content