import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from ollama import Client as OllamaClient
from duckduckgo_search import DDGS

# Initialize components
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection("kcc_farm_qa")
except:
    collection = client.create_collection("kcc_farm_qa")
ollama_client = OllamaClient()

def query_chromadb(user_query, top_k=3):
    biased_query = f"According to farming query: {user_query}"
    query_embedding = embedder.encode([biased_query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k * 2)
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    good_docs = [doc for doc, distance in zip(documents, distances) if distance < 0.8]
    return good_docs[:top_k]

def ask_model(prompt):
    response = ollama_client.chat(model='gemma3:1b', messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def fallback_search(query, max_results=2):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return [r["body"] for r in results]

def query_chatbot(user_query, top_k=3):
    docs = query_chromadb(user_query, top_k=top_k)
    if docs:
        st.markdown("- By Local DB")
        context = "\n\n".join([f"Context {i+1}: {d}" for i, d in enumerate(docs)])
        prompt = f"""
You are given some documents. First, **translate the following documents into English** and display them under the header **"Translations:"**

Then, using **only** the translated content and the provided context, write a **concise one-paragraph answer** to the question in **English**. Begin with a brief summary of all relevant context, then answer the question clearly and completely. Do not include any information not found in the documents or context.
---
Documents:
{docs}

Context:
{context}

Question:
{user_query}

---
Translations:
(Translate the documents in English here)

Answer:
(Write the answer here)
"""
    else:
        st.markdown("- By Internet Search")
        fallback = fallback_search(user_query)
        context = "\n\n".join([f"Web Result {i+1}: {d}" for i, d in enumerate(fallback)])
        prompt = f"""
Using only the provided context, write a **concise one-paragraph answer** to the question in **English**. Begin with a brief summary of the relevant context, then answer the question clearly and completely. Do not include any information that is not in the context.

---
Context:
{context}

Question:
{user_query}

---
Answer:
(Write the answer here)
"""
    try:
        return ask_model(prompt)
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("🌾 KCC Chatbot: Ask Farming Questions")

user_input = st.text_input("Enter your farming question:")
if user_input:
    with st.spinner("Thinking..."):
        reply = query_chatbot(user_input)
    st.markdown("### 🤖 Answer:")
    st.write(reply)

st.markdown("---")
st.markdown("**💡 Example Queries:**")
st.markdown("- What issues do sugarcane farmers in Maharashtra commonly face?")
st.markdown("- When will it rain in Maharashtra?")

