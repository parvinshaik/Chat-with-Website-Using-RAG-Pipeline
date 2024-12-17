import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Web scraping function with better chunking (splitting into sentences)
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.string
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    chunks = []
    for para in paragraphs:
        chunks.extend(para.split('.'))  # Split by sentences for better granularity
    return title, chunks

# Convert text to embeddings using SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(content):
    return model.encode(content)

# Create FAISS index with correct dimension (384)
index = faiss.IndexFlatL2(384)  # Adjusted for the 384-dimensional embeddings
data = {}

urls = ["https://www.uchicago.edu/", "https://www.washington.edu/", "https://www.stanford.edu/", "https://und.edu/"]

# Scrape websites and store embeddings in FAISS index
for url in urls:
    title, content_chunks = scrape_website(url)  # Use sentence chunks here
    embeddings = get_embeddings(content_chunks)  # Get embeddings for each chunk (sentence level)
    data[url] = {'title': title, 'content': content_chunks, 'embeddings': embeddings}
    
    embeddings_array = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)  # Normalize embeddings before adding to FAISS
    index.add(embeddings_array)

# Convert the query into an embedding
def query_to_embedding(query):
    return model.encode([query])

# Search function with better handling of indices
def search(query_embedding, index, k=5):
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)  # Normalize query embedding
    _, indices = index.search(query_embedding, k)
    
    # Check if indices[0] is empty or contains valid indices
    if indices.shape[1] == 0 or indices[0].size == 0:
        print("No relevant results found.")
        return []

    # Filter out indices that may be out of bounds for each URL's content
    retrieved_chunks = []
    
    for idx in indices[0]:
        # Ensure idx is a valid integer and within the range of available content for each URL
        if isinstance(idx, int):
            for url in data:
                if 0 <= idx < len(data[url]['content']):  # Ensure the index is within bounds
                    retrieved_chunks.append(data[url]['content'][idx])

    return retrieved_chunks

# Response generation using OpenAI
openai.api_key = "YOUR_API_KEY"  # Replace with your OpenAI API key

def generate_response(retrieved_chunks, query):
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the following question based on the context:\n\n{context}\n\nQuestion: {query}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()

# Example query
query = "What programs are offered at Stanford?"
query_embedding = query_to_embedding(query)

# Debug: print query embedding to inspect its values
print(f"Query embedding: {query_embedding}")

indices = search(query_embedding, index, k=8)  # Use k=8 to retrieve more candidates

if indices:
    # Get relevant content for response
    retrieved_chunks = []

    # Debug: check the retrieved indices and corresponding content
    print(f"Indices retrieved: {indices}")
    
    # Iterate over indices and filter by valid chunk length
    for idx in indices:
        if isinstance(idx, int):  # Check if the index is an integer
            for url in data:
                # Check if the index is within the valid content range for the URL
                if 0 <= idx < len(data[url]['content']):
                    retrieved_chunks.append(data[url]['content'][idx])

    if retrieved_chunks:
        response = generate_response(retrieved_chunks, query)
        print("Generated Response:")
        print(response)
    else:
        print("No relevant chunks found in retrieved indices.")
else:
    print("No relevant chunks were retrieved.")

