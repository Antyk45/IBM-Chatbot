import os
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from local import resolve
from config import options

def setup_llama():
    from langchain_community.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from llama_cpp.llama_cache import LlamaDiskCache
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path= os.getenv('LLAMA_MODEL_PATH'),
        # callback_manager = callback_manager,
        verbose = False,
        n_ctx=1024,
    )
    llm.client.set_cache(LlamaDiskCache())
    return llm

def setup_chatgpt():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature = 0.6)
    return llm

def setup_ai21():
    from langchain_community.llms import AI21
    llm = AI21(temperature=0)
    return llm

def get_raw_llm(llm_choice):
    global raw_llms
    llm_choice = llm_choice.lower()
    try:
        llm = raw_llms[llm_choice]
    except KeyError:
        if llm_choice == "llama":
            setup = setup_llama
        elif llm_choice == "ai21":
            setup = setup_ai21
        elif llm_choice == "chatgpt":
            setup = setup_chatgpt
        else:
            raise KeyError()
        llm = setup()
        raw_llms[llm_choice] = llm
    return raw_llms[llm_choice]
    
def get_llm(llm_choice):
    global llms
    llm_choice = llm_choice.lower()
    try:
        llm = llms[llm_choice]
    except KeyError:
        llm = get_raw_llm(llm_choice)
        system_prompt = resolve(options["language"], "system_prompt")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{system_prompt}\n\\{{context}}\n----------"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = RunnablePassthrough.assign(answer=document_chain)
        llms[llm_choice] = chain
    return llms[llm_choice]

llms = {}
raw_llms = {}
