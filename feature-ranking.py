import pymongo
from pymongo import MongoClient
import random

uri = "mongodb+srv://admin:wzE6nBcB4bnpyDUY@ibm.zbskp8h.mongodb.net/?retryWrites=true&w=majority&appName=IBM"

# Create a new client and connect to the server
client = MongoClient(uri)
db = client["IBM"]
collection = db["feature_ranking"]
num = random.random()

def addRating1(User_id, llm, rating):
    post = {"User_id" : User_id, "LLM": llm, "Rating" : rating}
    collection.insert_one(post)
    return

def addRating2(_id, User_id, llm1, rating1, llm2, rating2):
    post1 = {"_id" : _id, "User_id" : User_id, "LLM": llm1, "Rating" : rating1}
    post2 = {"_id" : _id, "User_id" : User_id, "LLM": llm2, "Rating" : rating2}
    collection.insert_one(post1)
    _id +=1
    collection.insert_one(post2)
    return

def addRating3( User_id, llm1, rating1, llm2, rating2, llm3, rating3):
    post1 = {"User_id" : User_id, "LLM": llm1, "Rating" : rating1}
    collection.insert_one(post1)
    post2 = { "User_id" : User_id, "LLM": llm2, "Rating" : rating2}
    collection.insert_one(post2)
    post3 = { "User_id" : User_id, "LLM": llm3, "Rating" : rating3}
    collection.insert_one(post3)
    return

def getAverageRating(LLM):
    results = collection.find({"LLM" : LLM}) 
    temp =0
    count =0
    for result in results:
        count = count + 1
        temp = temp + result["Rating"]
    
    return temp/count

def getUserRating(User_id, llm):
    result = collection.find_one({"User_id" :User_id,"LLM" : llm})
    return result["Rating"]

def getID():
    num = random.random()
    return num

addRating3( 1, "OpenAI", 2, "AI21", 5, "Llama", 3)
#addRating1(2, "OpenAI", 1)
#addRating1(3, "OpenAI", 2)
#addRating1(4, "OpenAI", 5)

#print(getUserRating(1, "AI21"))
#print(getAverageRating("OpenAI"))


