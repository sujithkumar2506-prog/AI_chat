from google import genai
from dotenv import load_dotenv
import numpy as np
import heapq
load_dotenv()
client = genai.Client()
def get_response(user_input=""):
    print("GPT starting...")
    system_instruction = "You are a RAG assistant and I will give you chunks and question. please read the chunks and respond to the question"
    model = "gemini-3.8-flash"
    if not user_input:
        user_input = "What is the difference between docker and kubernetes. explain it in brief words."
    interaction = client.interactions.create(
        model = model,
        input = user_input,
        system_instruction = system_instruction,
        store=False)
    # print(interaction)
    print(interaction.output_text)


def chunk_text_by_words(text,chunk_size=10,overlap=2):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be between 0 and chunk_size - 1")
    text = text.split('.')
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text),step):
        chunk = text[i:i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def chunk_text_by_sentences(text,chunk_size=10,overlap=5):
    sentences = text.split('.')
    chunks =[]
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        if current_length + sentence_word_count <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))  
    return chunks
            

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))    


def store_vectors(chunks):

    vector_store = dict()
    for i,chunk in enumerate(chunks):
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents= chunk
        )
        chunk_embedding = np.array(result.embeddings[0].values)
        vector_store[i+1] = {
            'context' : chunk,
            'embedding' : chunk_embedding
        }

    return vector_store

def embed_query(query):

    result = client.models.embed_content(
        model='gemini-embedding-2',
        contents = query
    )
    return result.embeddings[0].values


document = """
Company Leave Policy

Employees are entitled to 12 casual leave days annually.
Casual leave can be taken for personal or emergency purposes.

Employees are entitled to 15 sick leave days annually.
A medical certificate may be required for extended sick leave.

Special leave is limited to 6 days per year.
Special leave requires manager approval.

Employees must submit leave requests through the HR portal.
Requests should normally be submitted at least two days in advance.
"""


chunks = chunk_text_by_sentences(document, chunk_size=10, overlap=2)

vector_db = store_vectors(chunks)

def retrieve_chunks(query,vector_db, top_k=3):
    embedded_query = embed_query(query)
    embedded_query = np.array(embedded_query)
    scores = []
    for key in vector_db.keys():
        B = vector_db[key]['embedding']

        score = cosine_similarity(embedded_query,B)
        inverted_score = cosine_similarity(B,embedded_query)
        scores.append((score,key))
    scores.sort(key = lambda x: x[0])
    return scores[-1:-1-top_k:-1]

user_query = input('Enter your query: ')   
scores = retrieve_chunks(user_query,vector_db)
relevant_chunks = []
for score,key in scores:
    relevant_chunks.append(vector_db[key]['context'])
prompt = " Chunks are "
for chunk in relevant_chunks:
    prompt+=chunk+" "


prompt += "User question is "+user_query

get_response(prompt)