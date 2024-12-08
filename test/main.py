from ailite.langchain import ChatOpenAI
llm = ChatOpenAI()
print(llm.invoke("who are you?"))