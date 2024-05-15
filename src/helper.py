from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


#Extract data from the PDF
def load_pdf(data):
    logging.info(f"Loading PDF....")
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()
    logging.info(f"PDF loaded")
    return documents



#Create text chunks
def text_split(extracted_data):
    logging.info(f"Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    logging.info(f"text chunks are prepared.")
    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    logging.info(f"Loading Embedding model....")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info(f"Embedding model loaded")
    return embeddings



#download embedding model
def embedded_data(embeddings, text_chunks):
    logging.info(f"Initiating embedding for text chunks, please wait...")
    embeds = [embeddings.embed_query(t.page_content) for t in text_chunks]
    df = pd.DataFrame({'id': map(str, range(1, len(embeds)+1)), 'vectors': embeds})
    logging.info(f"Embedding completed for text chunks")
    return df