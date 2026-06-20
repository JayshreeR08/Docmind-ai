# 🪐 DocMind AI — Jupiter Station

DocMind AI is an intelligent, full-stack RAG (Retrieval-Augmented Generation) system built to solve a massive student problem: turning dense, exhausting exam notes into an interactive, conversational study room. 

Equipped with a custom cyber-glow dark theme, this application handles end-to-end document processing by combining localized semantic search vector layers with the reasoning capabilities of Google's advanced language models.

---

## 🚀 Instant Access (No Setup Required!)

You do not need to download any code, install Python, or configure anything to use this app. It is fully deployed in the cloud and ready for anyone to use instantly from a phone, tablet, or laptop.

👉 **[CLICK HERE TO USE DOCMIND AI LIVE](https://docmind-ai-jayshree081115jupcatrgs.streamlit.app/)**

### How to use the live app:
1. **Upload and Index:** Head over to the **Document Center** in the sidebar and upload your lecture notes or syllabus PDF. Click **🚀 Step 1: Index Document** to load it into the system memory.
2. **Extract Insights:** * Click **✨ Step 2: Auto-Summarize PDF** to instantly generate deep structural bullet points of your material.
   * Click **📝 Generate Practice Viva** to spin up an interactive mock exam station right above your workspace.
3. **Deep Drill-Down:** Use the main chat input bar at the bottom to ask specific questions about your notes or clear up confusing concepts.

---

## 🧠 Behind the Scenes: How Gemini Handles the Data

DocMind AI doesn't just read text; it understands context. When a document is uploaded, a highly coordinated data pipeline takes place:

1. **Document Splitting:** The system breaks long PDFs down into smaller, overlapping semantic chunks using a `RecursiveCharacterTextSplitter`.
2. **Vector Space Generation:** These chunks are converted into dense mathematical vectors using HuggingFace's `all-MiniLM-L6-v2` transformer model.
3. **The Search Cache:** When a user asks a question, the **FAISS** database finds the exact top 3 most relevant sections from the notes in milliseconds.
4. **Gemini Orchestration:** The app wraps those precise sections alongside the past chat history into a structured context window. This is sent to **`gemini-2.5-flash`**, which acts as the core reasoning engine—synthesizing accurate, hallucination-free technical answers directly from the material.

---

## ⚡ Core Features

* **🔮 Conversational Document Intelligence:** Chat seamlessly with your PDFs. If the answer isn't explicitly hidden within your uploaded material, the assistant will tell you directly, ensuring data accuracy.
* **✨ Dynamic Auto-Summarization:** Automatically analyzes complex document layers to output clean, bulleted summaries of core definitions and key concepts.
* **🎓 Practice Viva Mode:** The perfect exam-prep companion. It extracts high-density technical contexts to generate 5 challenging viva questions, complete with interactive, hidden click-to-reveal answer cards.
* **🌌 Cosmic Premium Aesthetic:** Custom CSS injections that completely override standard Streamlit layouts for a stunning midnight-blue space theme.

---

## 🛠️ Tech Stack

* **Orchestration Framework:** LangChain
* **LLM Engine:** Google Generative AI (`gemini-2.5-flash`)
* **Vector Database:** FAISS (CPU-optimized)
* **Embeddings:** HuggingFace Transformers (`all-MiniLM-L6-v2`)
* **Frontend Web UI:** Streamlit
* **PDF Parser Node:** PyMuPDF (Fitz)

---

## 💻 Developer Setup & Local Installation

If you are a developer and want to inspect the codebase, modify the architecture, or run this planetary base station locally on your desktop machine:

1.Clone the repository
2.Install the required dependencies:
   pip install -r requirements.txt
3.Configure your environment variables:
   Create a .env file in the root directory and add your key:
4.Ignite the local launch sequence:
    streamlit run app.py
