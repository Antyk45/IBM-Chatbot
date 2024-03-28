#UNIT TESTS

#Don't remove these imports
import ChatbotHelper
import app
import config
import os
import tempfile

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


from llm import setup_chatgpt, setup_ai21, setup_llama
from local import resolve
from csv_to_langchain import CSVLoader
from db import get_datafiniti_documents, populate_db

os.environ['LLM'] = 'TESTING'

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
    llm = setup_chatgpt()
    assert isinstance(llm, ChatOpenAI)
    del os.environ['LLM']

# def test_get_llm_LLAMA():                                     # Check that LLAMA is obtained correctly
#    os.environ['LLM'] = "LLAMA"                               # Load 'LLAMA' as the LLM selection
#    llm = setup_llama()
#    assert hasattr(llm, 'model_path')
#    assert llm.model_path == os.getenv('LLAMA_MODEL_PATH')
#    del os.environ['LLM']

def test_get_llm_AI21():                                     # Check that AI21 is obtained correctly
    os.environ['LLM'] = "AI21"                               # load 'AI21' as the LLM selection
    llm = setup_ai21()
    from langchain_community.llms import AI21
    assert isinstance(llm, AI21)
    del os.environ['LLM']

def test_get_memory():                                      # Check that the chat history is loaded correctly
    from langchain.memory import ChatMessageHistory
    assert ChatbotHelper.get_memory() == ChatMessageHistory()        

def test_get_memory_unique():                               # Check that calling the function multiple times does not change the result
    memory1 = ChatbotHelper.get_memory()
    memory2 = ChatbotHelper.get_memory()
    assert memory1 == memory2               
    


def test_csv_loader_can_load_and_rename_metadata():
   documents = get_datafiniti_documents("Test_Dataset.csv")
   assert len(documents) > 0, "no documents were loaded"
   for document in documents:
       assert document.metadata.get("rating") is not None, "rating not exposed as metadata"
       assert document.metadata.get("date") is not None, "date not exposed as metadata"

def test_document_persistence_works():
   with tempfile.TemporaryDirectory() as persist_directory:
       db = Chroma(
          persist_directory=persist_directory,
          embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
       )
       documents = get_datafiniti_documents("Test_Dataset.csv")
       r1 = populate_db(db, documents[:5])
       assert r1.existing_documents == 0, "no documents should exist in a fresh database"
       assert r1.new_documents == 5, "not all documents were added"
       r2 = populate_db(db, documents[:5])
       assert r2.existing_documents == 5, "all documents should exist due to the cache"
       assert r2.new_documents == 0, "despite no new documents being added, DB was mutated"
       r2 = populate_db(db, documents)
       assert r2.existing_documents == 5, "the first five documents we added were ignored"
       assert r2.new_documents == len(documents) - 5, "wrong number of documents added"


def test_metadata_update_without_reindex():
   """
   When we want to expose new metadata fields on a large amount of data, it
   is more efficient to adopt an approach that doesnt involve reindexing all the page contents.
   This test ensures that the populate_db function works like that
   """
   
   with tempfile.TemporaryDirectory() as persist_directory:
       db = Chroma(
          persist_directory=persist_directory,
          embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
       )
       documents = get_datafiniti_documents("Test_Dataset.csv")
       modified_documents = []
       for document in documents:
          metadata = { **document.metadata }
          del metadata["rating"]
          modified_document = Document(page_content=document.page_content, metadata=metadata)
          modified_documents.append(modified_document)
       r1 = populate_db(db, modified_documents)
       assert r1.updated_documents == 0, "no existing documents should have been updated"
       assert r1.existing_documents == 0, "no documents should exist in a fresh database"
       assert r1.new_documents == len(documents), "not all documents were added"
       for metadata in db.get()["metadatas"]:
          assert metadata.get("rating") is None, "rating metadata should not exist until later part of test!"
       r2 = populate_db(db, documents)
       assert r2.new_documents == 0, "no new documents should have been added in update operation"
       assert r2.updated_documents > 0, "no documents had their metadata updated"
       for metadata in db.get()["metadatas"]:
          assert metadata.get("rating") is not None, "metadata update was not successful!"
