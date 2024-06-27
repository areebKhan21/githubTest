import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import time

url = "https://www.whitehouse.gov/briefing-room/speeches-remarks/2024/03/07/remarks-of-president-joe-biden-state-of-the-union-address-as-prepared-for-delivery-2/"
query = "What did the president say about Ukraine?"

def fetch_web_page(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def split_text(text, chunk_size=40000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def create_embeddings(text_chunks, model="mxbai-embed-large"):
    embeddings = OllamaEmbeddings(model=model)
    return embeddings.embed_documents(text_chunks)

def store_embeddings_in_chroma(text_chunks, model="mxbai-embed-large"):
    docs = [Document(page_content=chunk, metadata={}) for chunk in text_chunks]
    db = Chroma.from_documents(docs, embedding_function=OllamaEmbeddings(model=model))
    return db

def query_chroma_db(db, query):
    return db.similarity_search(query)

def process_url_and_query(url, query):
    web_page_text = fetch_web_page(url)
    text_chunks = split_text(web_page_text)
    embeddings = create_embeddings(text_chunks)
    db = store_embeddings_in_chroma(text_chunks, embeddings)
    results = query_chroma_db(db, query)
    return results

result_docs = process_url_and_query(url, query)

for doc in result_docs:
    print(doc.page_content)
