import re
import sqlite3
import json
from langchain_core.messages import AIMessage
from langchain_core.language_models.llms import LLM
from llm import get_raw_llm
from db import get_db
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import Any, Dict, Iterator, List, Optional, Union, Callable
from dataclasses import dataclass

def inject_documents(connection: sqlite3.Connection):
    """
    Adds a "Reviews" table to the database, deleting any reviews table that exists already.
    """

    documents = get_db().get(include=["metadatas", "embeddings", "documents"])
    ids = documents["ids"]
    pages = documents["documents"]
    metadatas = documents["metadatas"]
    embeddings = documents["embeddings"]
    embeddings = { id : embedding for id, embedding in zip(ids, embeddings) }
    texts = { id : page for id, page in zip(ids, pages) }

    class SemanticMatch:
        def __init__(self, embeddings, texts):
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = embeddings
            self.texts = texts
            self.cache = {}
            
        def __call__(self, id, semantics):
            if semantics in self.texts[id]:
                return 1.0
            if semantics not in self.cache:
                self.cache[semantics] = self.model.encode(semantics)
            semantics_embedding = self.cache[semantics]
            document_embedding = self.embeddings[id]
            return cos_sim(semantics_embedding, document_embedding).item()

    connection.create_function("SEMANTIC_MATCH", 2, SemanticMatch(embeddings, texts), deterministic=True)
    connection.execute("DROP TABLE IF EXISTS reviews")
    connection.execute("""CREATE TABLE reviews(
Id string primary key,
Name varchar(255),
Rating integer,
Date varchar(255)
)
""")
    values = [(id, metadata["name"], metadata.get("rating") or '3', metadata.get("date"))
              for id, metadata in zip(ids, metadatas)]
    connection.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?)", values)
    connection.commit()

def explain_prompt(question, query, result):
    return f"""The "Reviews" database has the following fields

* "Id" - The unique ID of the review
* "Name" - The name of the product being reviewd
* "Rating" - Rating out of five
* "Date" - The date of the review in RFC3339 (e.g. the string "2024-02-04T00:00:00Z")

You will be given a question, the SQL generated for it, and the CSV result. You will have to create CSV where each string describes the corresponding result. Each element in the explanation MUST be a string containing a brief description. The explanation SHOULD NOT contain data.

--- (first example)

Query:
```sql
SELECT Name, AVG(Rating) GROUP BY Rating FROM Reviews;
```

Result:
```csv
"Kindle Paperwhite", "3.14"
```

Explanation:
```csv
"product name", "average rating"
```

--- (final example)

Query:
```sql
{query}
```

Result:
```csv
{",".join(json.dumps(str(r)) for r in result)}
```

Explanation (INCLUDE NO TEXT AFTER THE EXPLANATION):
```csv
"""

def initial_prompt(question):
    return f"""The "Reviews" database has the following fields

* "Id" - The unique ID of the review
* "Name" - The name of the product being reviewd
* "Rating" - Rating out of five
* "Date" - The date of the review in RFC3339 (e.g. the string "2024-02-04T00:00:00Z")

You will be given a question and will have to write a valid sqlite3 query using only the database and fields provided. Please follow the following guidelines.

* Using fields not listed above WILL LEAD TO MAJOR ERRORs.
* For matching on product names, emotions & concepts .etc, use the function `SEMANTIC_MATCH(Id, Concept)`. It returns a number in the range [0,1] representing how semantically similar the review body of the review is to the `Concept`. ALWAYS use this if the user asks about a specific product type. ONLY USE THIS FUNCTION WITH THE `Id` field.
* If you not doing an aggregation, ALWAYS return the Id for individual documents. Otherwise, the ID can be omitted.
* Give a concise description of each column of the result after the backticks. Exclude any and all SQL details. A good example is: "The first column describes the rating, the second column describes the date". This has no SQL details.

This is an example:

```sql
SELECT Id
FROM Reviews
WHERE Rating = 3 AND Name LIKE "%Kindle%" AND SEMANTIC_MATCH(Id, "Satisfied") > 0.3;
```

The question is: '{question}' (INCLUDE ONLY A BRIEF DESCRIPTION AFTER THE EXPLANATION).

```sql
"""

class QueryGenerator:
    def __init__(self, question: str, llm: LLM):
        self.question = question
        self.llm = llm

    def annotate(self, query: str, result: List[Any]) -> List[str]:
        """
        After a successful invocation of QueryGenerator.generate, use this to generate descriptions for
        each column returned in the result. This will give the LLM information about the question asked,
        the SQL query generated, and a row that was returned. It will use this data to derive column names
        for each column in the result.

        :param query: The SQL query
        :param result: A single row from the SQL query result
        """
        
        prompt = explain_prompt(question=self.question, result=result, query=query)
        answer = self.llm.invoke(prompt)
        answer = answer.content if type(answer) is AIMessage else answer
        answer = answer.split("```")[0]
        print("Annotation:", answer)
        return json.loads(f"[{answer}]" )
    
    def patch(self, query: str, error: str):
        """
        After an SQL query created by QueryGenerator.generate fails, use this to attempt to generate
        a query that actually works.

        :param query: The SQL query
        :param error: A message indicating the error that the LLM should try to fix
        """
        
        prompt = f"""{initial_prompt(question=self.question)}{query}
```

This doesn't work in sqlite3 due to the error: "{error}". A version that works is as follows:

```sql
"""
        answer = self.llm.invoke(prompt)
        answer = answer.content if type(answer) is AIMessage else answer
        answer = answer.split("```")[0]
        answer = answer.split(";")[0]
        select = re.search("select", answer, re.IGNORECASE)
        if select is None:
            raise RuntimeError("AI sure is unreliable (multiple times even!)")
        else:
            return answer[select.start():]
    
    def generate(self):
        """
        Takes a human question about reviews and generates an SQL query to assist in answering it.
        Basically lets the LLM query its own dataset
        """
        
        prompt = initial_prompt(question=self.question)
        answer = self.llm.invoke(prompt)
        answer = answer.content if type(answer) is AIMessage else answer
        try:
            answer, description = answer.split("```", 1)
            # strip the rest of the description in case LLM returns more than it should
            description, *_ = description.split("```", 1)
        except ValueError as e:
            description = ""
        answer = answer.split(";")[0]
        answer, description = answer.strip(), description.strip()
        select = re.search("SELECT", answer, re.IGNORECASE)
        if select is None:
            raise NoQueryGenerated("AI sure is unreliable!")
        else:
            return answer[select.start():], description

@dataclass
class AggregationRAGResult:
    """
    Represents the output from the AggregationRAG.answer function.
    """
    
    column_names: List[str]
    """
    The name of each column in the table
    """
    
    results: List[List[str]]
    """
    Columns returned by the SQL query. This is always non-empty.
    """
    
    description: str
    """
    An english description of what the results of the query generated should be.
    """
    
    query: str
    """
    The SQL query that ended up being generated to answer the question.
    """
    
    failed_attempts: List[str]
    """
    A list of the SQL queries that resulted in errors that the LLM attempted to fix.
    Only useful for debugging purposes.
    """

class LLMUnreliableException(Exception):
    pass

class NoQueryGenerated(LLMUnreliableException):
    pass

class AggregationRAG:
    """
    This attempts to use SQL based RAG to answer aggregation questions about the product review
    databse as a whole
    """
        
    def __init__(
            self,
            llm: LLM,
            verbose: bool =False,
            notify_cb: Callable[[str], None] = lambda _: None
    ):
        self.llm = llm
        self.connection = sqlite3.connect(":memory:")
        self.verbose = verbose
        self.notify_cb = notify_cb
        inject_documents(self.connection)
        
    def answer(self, question: str) -> Optional[AggregationRAGResult]:
        """
        Attempt to answer the question by accessing the global document database.
        
        :param question: The question
        """
        
        self.notify_cb("Attempting to generate SQL…")
        query_generator = QueryGenerator(question, self.llm)
        failed_attempts = []
        attempts = 3
        error, fix = None, None
        for attempt in range(attempts):
            if fix is not None:
                answer = query_generator.patch(fix, error)
            else:
                try:
                    answer, description = query_generator.generate()
                except NoQueryGenerated:
                    continue
            if self.verbose:
                print("SQL Query:", answer)
                print("Description:", description)
            try:
                self.notify_cb("Performing query…")
                results = list(self.connection.execute(answer))
                if len(results) > 0:
                    self.notify_cb("Annotating generated table…")
                    annotation = query_generator.annotate(answer, results[0])
                    return AggregationRAGResult(
                        column_names = annotation,
                        results = results,
                        description = description,
                        query = answer,
                        failed_attempts = failed_attempts
                    )
                else:
                    return None
            except Exception as e:
                if self.verbose:
                    print("Repeating due to error:", e)
                error, fix = str(e), answer
                failed_attempts.append(fix)
        else:
            raise LLMUnreliableException("LLM unreliable 🗿!")

if __name__ == "__main__":
    llm = get_raw_llm("ai21")
    agg = AggregationRAG(llm)
    
    print("Finished setting up database")
    for question in [
            "How many reviews talk about batteries?",
            "How many reviews are angry?",
            "How many reviews are delighted?",
            "How many reviews were about batteries?",
            "Which batteries received ratings above 4?",
            "How many reviews were created in 2023?",
            "Give me the count of reviews created in 2021?",
            "What amount of reviews were made in 2017?",
            "What were the 10 most popular products?",
            "What was the average rating for each product?",
            "What was the lowest rated product?",
            "What was the highest rated product?",
            "On what date was the most recent review?"
    ]:
        print("---")
        print("QUESTION:", question)
        try:
            answer = agg.answer(question)
            if answer is not None:
                print(json.dumps(answer.__dict__, indent=4))
        except LLMUnreliableException as e:
            print(f"Unable to generate shit: {e}")
        print("---\n")
