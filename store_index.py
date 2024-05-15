from src.helper import *
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import random
import itertools
import pandas


# Load environment variables
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

# Setting up the embedding dataframe
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
df = embedded_data(embeddings, text_chunks)

# Check if the API key is loaded
if not API_KEY:
    raise ValueError("Pinecone API key not found. Please check your environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=API_KEY)

# Define index name
index_name = "test-my-code"

# Verify the index exists
indexes = pc.list_indexes()
if index_name not in indexes.names():
    # Create index if it doesn't exist (optional)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    logging.info(f"New index ready as {index_name}")

# Connect to the index
index = pc.Index(index_name)

# Convert DataFrame to list of (id, vector) tuples
data_tuples = list(df.itertuples(index=False, name=None))

# Helper function to break an iterable into chunks of size batch_size
def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# Upsert data with 100 vectors per upsert request
logging.info(f"Pushing embedded data to {index_name}")
for ids_vectors_chunk in chunks(data_tuples, batch_size=100):
    index.upsert(vectors=ids_vectors_chunk)