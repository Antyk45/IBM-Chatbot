import os.path

# Document loading and the link
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever

# ✨AI✨
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, SystemMessage

# Our own stuff
from csv_to_langchain import CSVLoader
from local import resolve

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, field
import datetime
from dotenv import load_dotenv, find_dotenv, dotenv_values

from agent.sql import AggregationRAG, LLMUnreliableException
from db import get_db
from llm import get_llm, get_raw_llm

@dataclass
class PendingInferenceComplete:
    data: object
    timestamp: str = field(default_factory=lambda: str(datetime.datetime.now()))
      
@dataclass
class PendingResponseChoice:
    answers: Dict[str, str]
    """
    A mapping from llm names → answers
    """

def query_chain(retriever):
    return (lambda params: params["messages"][-1].content) | retriever

def set_state(next_state):
    global state
    state = next_state
    print(f"Transitioned to next state {state}")

def get_state():
    global state
    return state

@dataclass
class Closure:
    llm: str
    def __call__(self, part, whole):
        print(f"got message from `{self.llm}`: {part}")
        socketio.emit("socket", {
            "llm": self.llm,
            "answer": whole.get("answer", "")
        })

def _get_data(messages, llm_choices, sql=False):
    jobs = []
    pool = ThreadPoolExecutor(4)

    question = messages[-1].content
    context = get_db().as_retriever(k=1).invoke(question)
    context_source = RunnablePassthrough.assign(context=lambda _: context)
    
    for llm_choice in llm_choices:
        llm = get_raw_llm(llm_choice)
        chain = context_source | get_llm(llm_choice)
        print(f"Submitting task for `{llm_choice}`")
        job = pool.submit(infer, messages, llm, chain, Closure(llm_choice), sql=sql)
        jobs.append((llm_choice, job))

    answers = []
        
    for llm, job in jobs:
        answer = job.result()
        answers.append({ **answer, "llm": llm })
    
    return answers, context

def infer(messages, llm, chain, callback, sql=False):
    if sql:
        print("Running pre-inference step")
        try:
            agg = AggregationRAG(llm, verbose=True, notify_cb=lambda event: callback({}, { "answer" : event }))
            result = agg.answer(messages[-1])
            if result is not None:
                full = { **result.__dict__, "type" : "tabular" }
                return full
            else:
                print("got empty result, but things were otherwise okay")
        except LLMUnreliableException as e:
            print(f"LLM not reliable: {e}")
    else:
        print("SQL Aggregation tool disabled")
        
    print("Starting inference for LLM")
    payload = {
        "messages": [
            *messages,
            SystemMessage(content="No tabular output could be generated. Use the sources provided to answer the question. Note that these sources are limited.")
        ]
    }
    full = None
    for item in chain.stream(payload):
        if full is None:
            full = item
        else:
            full += item
        if callback is not None:
            callback(item, full)
    return { "type" : "regular", "answer" : full["answer"] }

set_state(None)
memory = ChatMessageHistory()


def refresh_environment_vars():
    for env_key, env_value in dotenv_values('.env').items():
        os.environ[env_key] = env_value

if __name__ == "__main__":
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from flask_socketio import SocketIO
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    CORS(app)
    path_to_env = find_dotenv()
    if path_to_env:
        load_dotenv(path_to_env)
    else:
        open('.env', 'w')
        load_dotenv('.env')
    refresh_environment_vars()

    @app.route('/message', methods=['POST'])
    def get_data():
        global state
        if state is not None:
            return jsonify(
                error="Already running inference",
                state=str(state)
            ), 400
        data = request.json
        llm_choices = data.get('llms') or []
        if len(llm_choices) == 0:
            return jsonify(answers={})
        message = data.get('message')
        sql = data.get('sql')
        set_state(PendingInferenceComplete(data=data))
        memory.add_user_message(message)
        answers, sources = _get_data(memory.messages, llm_choices, sql=sql)
        set_state(PendingResponseChoice(answers={ answer["llm"]: answer for answer in answers }))
        return jsonify({
            "answers": answers,
            "sources" : [ { "pageContent" : source.page_content,
                            "title" : source.metadata["title"],
                            "rating" : source.metadata["rating"],
                            "productName" : source.metadata["name"] }
                          for source in sources ]
        })
    
    
    @app.route('/selectAnswer', methods=['POST'])
    def select_response():
        data = request.json
        global state
        if type(state) is not PendingResponseChoice:
            return jsonify(400)
        chosen_answer = data.get('llm')
        global memory
        chosen_answer = state.answers[chosen_answer.lower()]
        if chosen_answer["type"] == "tabular":
            memory.add_ai_message("[This question was answered in a tabular format, which was presented in the UI]")
        else:
            memory.add_ai_message(chosen_answer["answer"])
        set_state(None)
        return jsonify(200)


    @app.route('/config', methods=['POST'])
    def config_llm():
        data = request.json
        global state
        llm = data.get('llm')
        llm_key = data.get('llm_key')
        with open('.env', 'r') as read_dotenv:
            lines = read_dotenv.readlines()
        lines_to_write = [f'{llm}={llm_key}']
        with open('.env', 'w') as write_dotenv:
            for line in lines:
                if not (line.startswith(llm) or line.isspace()):
                    lines_to_write.append(f'\n{line}')
            write_dotenv.writelines(lines_to_write)
        load_dotenv(path_to_env)
        refresh_environment_vars()
        set_state(None)
        return jsonify(200)

    app.run(port=5000)
