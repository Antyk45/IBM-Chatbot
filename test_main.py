#UNIT TESTS

#Don't remove these imports
import ChatbotHelper
import app
import config
import os

# Document loading and the link
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# ✨AI✨
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain


from local import resolve
from csv_to_langchain import CSVLoader

#Following code, that is commented out can only be run with a local machine
#You need to provide AI21 key, OPENAI key, LLAMA path in config.py


# def test_get_db():
#
# # Test 1 - LLM set to TESTING and creating new embedding
#     os.environ['LLM'] = 'TESTING'
#     documents = CSVLoader("Test_Dataset.csv").load()[:1]
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
#     split_documents = text_splitter.split_documents(documents)
#  # Create instances of ChatbotHelper.get_db()
#     db = Chroma.from_documents(
#             split_documents,
#             SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
#             ids=[str(i) for i in range(len(split_documents))],
#             persist_directory=".chroma_db"
#         )
#
#     db1 = ChatbotHelper.get_db()
#  # Perform similarity search
#     docs = db.similarity_search("Kindle battery")
#     docs1 = db1.similarity_search("Kindle battery")
#
#  # Check if the documents have the same length
#     assert len(docs) == len(docs1)
#
#  # Check page_content for each document
#     for i in range(len(docs)):
#         assert docs[i].page_content == docs1[i].page_content
#
#
#  # Test 2 - LLM set to Testing and chroma db exists
#  # Create instances of ChatbotHelper.get_db()
#     db = Chroma(
#             persist_directory=".chroma_db",
#             embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#         )
#     db1 = ChatbotHelper.get_db()
#
#  # Perform similarity search
#     docs = db.similarity_search("Kindle battery")
#     docs1 = db1.similarity_search("Kindle battery")
#
#  # Check if the documents have the same length
#     assert len(docs) == len(docs1)
#
#  # Check page_content for each document
#     for i in range(len(docs)):
#         assert docs[i].page_content == docs1[i].page_content

# # Test 3 - for LLM environ not set to TESTING and create new embeddings
#     os.environ['LLM'] = 'AI21'
#     documents = CSVLoader("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv").load()[:10] # take the first 10 rows
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
#     split_documents = text_splitter.split_documents(documents)
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#
#  # Create instances of ChatbotHelper.get_db()
#     db = Chroma.from_documents(
#             split_documents,
#             SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
#             ids=[str(i) for i in range(len(split_documents))],
#             persist_directory=".chroma_db"
#         )
#     db1 = ChatbotHelper.get_db()
#
#  # Perform similarity search
#     docs = db.similarity_search("Kindle battery")
#     docs1 = db1.similarity_search("Kindle battery")
#
#  # Check if the documents have the same length
#     assert len(docs) == len(docs1)
#
#  # Check page_content for each document
#     for i in range(len(docs)):
#         assert docs[i].page_content == docs1[i].page_content
#
# # Test 4 - LLM environment not set to Testing and chromedb exists
#     os.environ['LLM'] = 'AI21'
#  # Create instances of ChatbotHelper.get_db()
#     db = Chroma(
#             persist_directory=".chroma_db",
#             embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#         )
#     db1 = ChatbotHelper.get_db()
#
#  # Perform similarity search
#     docs = db.similarity_search("Kindle battery")
#     docs1 = db1.similarity_search("Kindle battery")
#
#  # Check if the documents have the same length
#     assert len(docs) == len(docs1)
#
#  # Check page_content for each document
#     for i in range(len(docs)):
#         assert docs[i].page_content == docs1[i].page_content
#

def test_get_options():
   assert ChatbotHelper.get_options() == {"language" : "English"}


def test_get_llm_ChatOpenAI():                                  # Check that ChatOpenAI is obtained correctly
    os.environ['LLM'] = "CHATGPT"                               # Load 'CHATGPT' as the LLM selection
    from langchain_openai import ChatOpenAI
    llm = app.setup_chatgpt()
    assert isinstance(llm, ChatOpenAI)
    del os.environ['LLM']

# def test_get_llm_LLAMA():                                     # Check that LLAMA is obtained correctly
#    os.environ['LLM'] = "LLAMA"                               # Load 'LLAMA' as the LLM selection
#    llm = app.setup_llama()
#    assert hasattr(llm, 'model_path')
#    assert llm.model_path == os.getenv('LLAMA_MODEL_PATH')
#    del os.environ['LLM']

def test_get_llm_AI21():                                     # Check that AI21 is obtained correctly
    os.environ['LLM'] = "AI21"                               # load 'AI21' as the LLM selection
    llm = app.setup_ai21()
    from langchain.llms import AI21
    assert isinstance(llm, AI21)
    del os.environ['LLM']

def test_get_memory():                                      # Check that the chat history is loaded correctly
    from langchain.memory import ChatMessageHistory
    assert ChatbotHelper.get_memory() == ChatMessageHistory()        

def test_get_memory_unique():                               # Check that calling the function multiple times does not change the result
    memory1 = ChatbotHelper.get_memory()
    memory2 = ChatbotHelper.get_memory()
    assert memory1 == memory2               
    


