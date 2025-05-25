# Sharvari_Raut_KCCQueryAssistant


# 🌾 KCC Chatbot: Farming Q&A with ChromaDB, MPNet & Gemma

This chatbot system helps answer agricultural questions using a vector store (ChromaDB), sentence embeddings (MPNet), and an LLM (Gemma via Ollama). It has three key components: preprocessing, embedding, and a Streamlit-based UI for user interaction.

---

## 📦 Features

- Language detection and cleaning of text
- Embedding question-answer data using `all-mpnet-base-v2`
- Vector storage using ChromaDB
- Context-aware response generation with `gemma:3b` via Ollama
- Web fallback using DuckDuckGo if vector search fails
- Streamlit interface

---

## 🏗️ Project Structure

|- preprocess.py # Clean, detect language, and chunk CSV

|- embedder.ipynb # Embed text into ChromaDB

|- Streamlit_processor.py # Main chatbot with UI and logic

|- queries.csv # Your Q&A dataset

|- chroma_db/ # Vector store folder

|- README.md # This file


---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-org/kcc-chatbot.git
cd kcc-chatbot

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

🧼 Preprocessing Script (Preprocessing Data.py)

This script:

    Cleans text (removes HTML, normalizes whitespace)

    Detects language and filters English-only rows

    Determines if each row is a Q and A or document

    Splits content into chunks for embedding

🧠 Embedding Script (Embedder.ipynb)

This notebook:

    Loads queries.csv

    Cleans and combines questions/answers

    Embeds using MPNet (all-mpnet-base-v2)

    Stores data in ChromaDB using batched inserts

🧠 Main Processor (Streamlit_processor.py)
Responsibilities:

    Accepts user question

    Embeds and searches vector DB

    If match: answers from local DB

    If no match: searches web (DuckDuckGo)

    Uses gemma:3b from Ollama for final answer

    Streamlit frontend for interaction

# Start Streamlit app to run the chatbot
```
streamlit run Streamlit_processor.py
```
Go to http://localhost:8501 in your browser.


💬 Example Questions

    What issues do sugarcane farmers in Maharashtra face?

    How to treat whiteflies in tomatoes?

    When will it rain in Solapur?

    Which crops grow best in black soil?


📌 Notes

    The ChromaDB vector limit is 5461 embeddings per request. Batching is handled in the embedder script.

    Fallback uses DuckDuckGo to provide minimal info when vector search fails.

    Translation logic is part of the Gemma prompt, not a separate service.
