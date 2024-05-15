from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
# Run once from terminal: pip install --upgrade langchain pinecone-client


# Initializing Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

# Defining embedding model
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = Pinecone(api_key=API_KEY)

# Define index name
index_name = "test-my-code"

#Loading the index
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="Llama2-model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})



# Instantiate the RetrievalQA chain
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
#                                  retriever=docsearch.as_retriever(), 
#                                  chain_type_kwargs=chain_type_kwargs)




@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)