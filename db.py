import functools
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

from csv_to_langchain import CSVLoader
from typing import Any, Dict, Iterator, List, Optional, Union

import config

def get_datafiniti_documents(file_name: str) -> List[Document]:
    loader = CSVLoader(
       file_name,
       metadata_columns=["reviews.rating", "reviews.date", "reviews.title"],
       metadata_column_names=["rating", "date", "title"]
    )
    documents = loader.load()
    return documents

@functools.cache
def get_db():
    documents = get_datafiniti_documents("_Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
    print("Loading persisted ChromaDB data store")
    db = Chroma(
        persist_directory=".chroma_db",
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    populate_db(db, documents)
    return db

@dataclass
class PopulateDBResult:
    updated_documents: int
    """
    The number of documents that had to have their metadata updated
    """
    
    new_documents: int
    """
    The number of documents that had to be added.
    """
    
    existing_documents: int
    """
    The number of documents that existed already in the database.
    """

def populate_db(db: Chroma, documents: List[Document]) -> PopulateDBResult:
    persisted_documents = db.get()

    print("Figuring out which documents exist already")
    existing_documents = set((metadata["source"], metadata["row"]) for metadata in persisted_documents["metadatas"])
    document_map = {(document.metadata["source"], document.metadata["row"]) : document for document in documents}
    new_documents = [document_map[key] for key in set(document_map.keys()).difference(existing_documents)]
    
    new_metadata_ids = []
    new_metadata_metadatas = []
    for id, metadata in zip(persisted_documents["ids"], persisted_documents["metadatas"]):
        key = (metadata["source"], metadata["row"])
        try:
            parent_document = document_map[key]
            if parent_document.metadata != metadata:
                new_metadata_ids.append(id)
                new_metadata_metadatas.append(parent_document.metadata)
        except KeyError:
            print(f"document {key} does not have an invalid parent, not considering for metadata update")
            continue

    if len(new_metadata_ids) > 0:
        print(f"Need to update the metadata for {len(new_metadata_ids)} documents")
        if config.FEEDBOT_IGNORE_NEW_DOCUMENTS:
            print("\tFEEDBOT_IGNORE_NEW_DOCUMENTS is set, skipping…")
        else:
            i = 0
            while i < len(new_metadata_ids):
                db._client \
                  .get_collection("langchain") \
                  .update(ids=new_metadata_ids[i:i+5000],
                          metadatas=new_metadata_metadatas[i:i+5000])
                i += 5000
                print(f"Hit {i} document mark…")
    else:
        print(f"No existing document metadata needs to be changed")

    if len(new_documents) > 0:
        print(f"Need to split and add {len(new_documents)} to ChomaDB")
        if config.FEEDBOT_IGNORE_NEW_DOCUMENTS:
            print("\tFEEDBOT_IGNORE_NEW_DOCUMENTS is set, skipping…")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
            split_documents = text_splitter.split_documents(new_documents)
            print(f"Split {len(new_documents)} documents. Adding to Chromadb…")
            i = 0
            while i < len(split_documents):
                db.add_documents(split_documents[i:i+5000])
                i += 5000
                print(f"Hit {i} document mark…")
            print(f"Added {len(new_documents)} documents to ChromaDB")
    else:
        print(f"All {len(documents)} are already in the database")
    return PopulateDBResult(
        updated_documents=len(new_metadata_ids),
        new_documents=len(new_documents),
        existing_documents=len(existing_documents)
    )
