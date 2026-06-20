
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
import os
import subprocess

# --- AUTOMATIC DEPENDENCY CHECKER ---
required_libraries = {
    "streamlit": "streamlit",
    "langchain_chroma": "langchain-chroma",
    "langchain_community": "langchain-community",
    "pypdf": "pypdf",
    "dotenv": "python-dotenv",
    "langchain_google_genai": "langchain-google-genai",
    "fitz": "pymupdf",
    "torchvision": "torchvision"
}

for module, package in required_libraries.items():
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --- CLEAN IMPORTS ---
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="DocMind AI | Jupiter Station", layout="centered")

# --- PREMIUM PORTFOLIO DESIGN THEME ENGINE ---
st.markdown("""
    <style>
    /* Portfolio Midnight Blue Dark Canvas Background */
    .stApp { 
        background: radial-gradient(circle at top, #0c101b, #05070c) !important;
    }
    
    /* Text Color Normalization overrides */
    p, span, label, div { color: #CBD5E1 !important; }
    
    /* Cyber-Glow Main Title Text Header */
    h1 { 
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Courier New', Courier, monospace;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px !important;
    }
    
    /* Dynamic Interactive Subheading Banner */
    .tagline {
        text-align: center;
        color: #38BDF8 !important;
        font-size: 1.1rem;
        margin-bottom: 25px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Layout Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1321 !important;
        border-right: 1px solid rgba(79, 172, 254, 0.2);
    }
    
    /* Cosmic Custom Sidebar Brand Container Grid Box */
    .cosmic-box {
        background: linear-gradient(145deg, rgba(13,19,33,0.8), rgba(20,30,55,0.8));
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.05);
    }
    .space-grid {
        font-size: 1.8rem;
        letter-spacing: 8px;
        margin-top: 8px;
        display: block;
    }

    /* Premium Neon-Border Chat Message Containers */
    .stChatMessage { 
        border-radius: 16px; 
        background-color: #0f172a !important; 
        border: 1px solid rgba(79, 172, 254, 0.15);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); 
        margin-bottom: 15px; 
    }
    
    /* Action Buttons Custom Look Configuration */
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #05070c !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100%;
        margin-bottom: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
    }
    
    /* Custom spacing for expanding answers */
    .stMarkdown div[data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        border-radius: 8px;
    }
    
    /* Creative Signature Badge Footer */
    .portfolio-footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .heart-glow {
        color: #00f2fe;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(0, 242, 254, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# Main Screen App Interface Header Elements
st.title("DocMind AI")
st.markdown('<p class="tagline">🪐 Conversational Document Intelligence Active (Broadcasting from Jupiter).</p>', unsafe_allow_html=True)

DB_DIR = "chroma_db"

@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

vector_store = get_vector_store()

if "current_chunks" not in st.session_state:
    st.session_state.current_chunks = []
if "viva_questions" not in st.session_state:
    st.session_state.viva_questions = None

# --- FREE GEMINI LLM INITIALIZATION ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload your notes (PDF):", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Step 1: Index Document"):
            with st.spinner("Processing text nodes..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(docs)
                    
                    vector_store.add_documents(chunks)
                    st.session_state.current_chunks = [c.page_content for c in chunks]
                    st.session_state.viva_questions = None  # Reset old questions for new file
                    st.success(f"📦 Document Loaded Successfully!")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
                finally:
                    os.remove(tmp_path)
        
        if len(st.session_state.current_chunks) > 0:
            if st.button("✨ Step 2: Auto-Summarize PDF"):
                with st.spinner("Analyzing document structure..."):
                    sample_context = "\n\n".join(st.session_state.current_chunks[:5])
                    summary_prompt = (
                        f"Provide a comprehensive, clear, and well-structured professional summary of the following text "
                        f"using bullet points for key concepts:\n\n{sample_context}"
                    )
                    try:
                        res = llm.invoke(summary_prompt)
                        st.session_state.messages.append({"role": "assistant", "content": f"📝 **Here is the summary of your uploaded document:**\n\n{res.content}"})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gemini API Error: {e}")

            # --- VIVA FEATURE BUTTON ---
            if st.button("📝 Generate Practice Viva"):
                with st.spinner("Formulating exam questions..."):
                    viva_context = "\n\n".join(st.session_state.current_chunks[:6])
                    viva_prompt = (
                        f"Based on the following technical context text, generate 5 challenging, descriptive exam or viva questions.\n"
                        f"Format your response exactly like this string format so I can parse it cleanly:\n"
                        f"Q1: Question text here\nA1: Answer text here\n"
                        f"Q2: Question text here\nA2: Answer text here\n"
                        f"Context data:\n{viva_context}"
                    )
                    try:
                        res = llm.invoke(viva_prompt)
                        st.session_state.viva_questions = res.content
                    except Exception as e:
                        st.error(f"Error generating quiz: {e}")

    # --- BRANDED JUPITER EASTER EGG SIDEBAR SECTION ---
    st.markdown("""
        <div class="cosmic-box">
            <span style="font-weight: bold; color: #00f2fe !important;">🚀 Jupiter Base Station</span>
            <div class="space-grid">🪐 👨‍🚀 🛰️ 🌌 🔭</div>
            <p style="font-size: 0.8rem; color: #64748B !important; margin-top: 5px;">Analyzing data layers across the system...</p>
        </div>
    """, unsafe_allow_html=True)

# --- RENDER VIVA QUESTIONS ABOVE CHAT IF ACTIVE ---
if st.session_state.viva_questions:
    st.markdown("### 🎓 Interactive Practice Viva Mode")
    st.markdown("Test your concepts before the exam! Click the dropdowns to reveal answers.")
    
    raw_text = st.session_state.viva_questions
    lines = raw_text.split('\n')
    current_q = ""
    
    for line in lines:
        if line.startswith("Q") and ":" in line:
            current_q = line.split(":", 1)[1].strip()
        elif line.startswith("A") and ":" in line and current_q:
            current_a = line.split(":", 1)[1].strip()
            with st.expander(f"❓ {current_q}"):
                st.markdown(f"💡 **Correct Viva Answer:**\n{current_a}")
            current_q = ""
    st.markdown("---")

# SYSTEM PROMPT DESIGN WITH EXPLICIT HISTORY ENGINE LAYER
system_prompt = ChatPromptTemplate.from_template(
    "You are DocMind AI, a specialized question-answering assistant.\n"
    "Use the following fragments of retrieved context from uploaded documents to answer user queries.\n"
    "You also have access to the conversation history below to maintain continuity across multi-turn exchanges.\n"
    "If the answer cannot be confidently inferred from the context data, explicitly state "
    "that you do not possess information on that topic within your current repository.\n\n"
    "--- CONVERSATION HISTORY ---\n"
    "{chat_history}\n\n"
    "--- RETRIEVED TEXT CONTEXT ---\n"
    "{context}\n\n"
    "User Question: {question}"
)

# --- CHAT INTERFACE LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🦊" if msg["role"] == "user" else "🐼"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask DocMind AI anything about your saved notes..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🦊"):
        st.markdown(user_input)
        
    with st.chat_message("assistant", avatar="🐼"):
        with st.spinner("Searching database layers..."):
            try:
                history_str = ""
                for msg in st.session_state.messages[-7:-1]:
                    role_label = "Human" if msg["role"] == "user" else "Assistant"
                    history_str += f"{role_label}: {msg['content']}\n"
                if not history_str:
                    history_str = "No prior exchanges in current active tracking queue."

                docs = vector_store.similarity_search(user_input, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                formatted_prompt = system_prompt.format(
                    chat_history=history_str, 
                    context=context, 
                    question=user_input
                )
                response = llm.invoke(formatted_prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# --- CREATIVE PORTFOLIO SIGNATURE FOOTER ---
st.markdown("""
    <div class="portfolio-footer">
        ⚡ Made by <span class="heart-glow">🪐</span> <span class="heart-glow">Jayshree</span> | Direct from the Great Red Spot
    </div>
""", unsafe_allow_html=True)
