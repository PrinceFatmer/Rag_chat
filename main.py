import streamlit as st
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from openai import OpenAI

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

client = OpenAI(api_key=api_key)

# Page settings
st.set_page_config(
    page_title="📄 Bring Document to Life",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📄 Bring Document to Life</h1>", unsafe_allow_html=True)
st.markdown("### Upload a PDF and transform it into a searchable, intelligent knowledge base 💡")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False
if "ingested" not in st.session_state:
    st.session_state.ingested = False
# Upload section
with st.container():
    uploaded_file = st.file_uploader("📤 Upload your PDF file here", type=["pdf"])

    if uploaded_file:
        # Generate a simple hash based on file name and size to track changes
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if it's a new file
        if st.session_state.get("last_file_id") != file_id:
            st.session_state.last_file_id = file_id
            st.session_state.chat_ready = False
            st.session_state.ingested = False
            st.session_state.messages = []

        if not st.session_state.ingested:
            with st.spinner("🔍 Reading and extracting PDF content..."):
                pdf_bytes = uploaded_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                docs = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"page": page_num + 1}))

                st.success(f"✅ Extracted `{len(docs)}` pages from the document.")

            with st.spinner("✂️ Splitting content into manageable chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
                split_docs = text_splitter.split_documents(docs)
                st.success(f"📚 Split into `{len(split_docs)}` text chunks.")

            with st.spinner("🔗 Generating embeddings and saving to vector store..."):
                embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",  google_api_key=gemini_key)
                split_docs = [doc for doc in split_docs if doc.page_content.strip()]
                if not split_docs:
                    st.error("❌ No valid text found in the PDF to embed. Please upload a different file.")
                    st.stop()
                vector_store = Qdrant.from_documents(
                    documents=split_docs,
                    embedding=embedding_model,
                    url="http://localhost:6333",  # Adjust as needed
                    collection_name="learning_vectors"
                )

            st.success("✅ Document successfully embedded and indexed.")
            st.balloons()
            st.session_state.chat_ready = True
            st.session_state.ingested = True
    else:
        st.info("📄 Please upload a PDF file to get started.")


# Optional: Sidebar
st.sidebar.markdown("## 📘 Instructions")
st.sidebar.markdown(
    """
1. Upload a PDF file.
2. Wait for content extraction and indexing.
3. Use the chat (to be implemented) to interact with the document.
"""
)

st.sidebar.markdown("💡 *This app uses Google Gemini embeddings and Qdrant vector DB.*")

if st.session_state.chat_ready:
    st.title("🤖 Your Document is ready to chat")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",  google_api_key=gemini_key)
    vector_db = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        embedding=embedding_model,
        collection_name="learning_vectors"
    )
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about the PDF...")
    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role":"user", "content":user_query})

        # Put all heavy work inside spinner context
        with st.spinner("Searching..."):
            results = vector_db.similarity_search(query=user_query)
            
            if results:
                context = "\n\n".join([doc.page_content for doc in results])
                pages = ", ".join([str(doc.metadata.get("page", "N/A")) for doc in results])

                SYSTEM_PROMPT = f"""
                You are a helpful AI Assistant who answers user queries based on the available context
                retrieved from a PDF file along with page_contents and page numbers.

                You should only answer the user based on the following context and guide the user
                to open the right page number to know more.

                Context:
                {context}

                If the answer is not in the context, say "I couldn’t find the answer in the PDF."
                """

                try:
                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[
                            { "role": "system", "content": SYSTEM_PROMPT },
                            { "role": "user", "content": user_query },
                        ]
                    )
                    answer = response.choices[0].message.content

                    if answer != "I couldn’t find the answer in the PDF.":
                        reply = f"{answer}\n\n📄 *Based on pages: {pages}*"
                    else:
                        reply = f"{answer}\n\n📄"
                except Exception as e:
                    reply = f"⚠️ Error with LLM summarization: {e}\n\nFallback:\n{results[0].page_content}\n📄 Page: {pages}"
            else:
                reply = "❌ I couldn’t find a relevant answer."

        # Display the assistant's reply (outside spinner so UI updates properly)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
