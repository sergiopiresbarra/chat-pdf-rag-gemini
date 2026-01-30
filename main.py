import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import time
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- IMPORTAÇÕES DA VERSÃO ESTÁVEL (v0.1.20) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega variáveis
load_dotenv()

st.set_page_config(page_title="Chat com PDF - RAG", page_icon="🤖")
st.title("🤖 Chat com seus Documentos (RAG)")

with st.sidebar:
    st.header("Upload do Arquivo")
    uploaded_file = st.file_uploader("Envie seu PDF", type="pdf")
    api_key = st.text_input("Cole sua Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

def process_pdf(uploaded_file):
    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Cria pedaços MAIORES para fazer MENOS requisições
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # --- MUDANÇA 1: Modelo mais novo ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # --- MUDANÇA 2: Estratégia "Tartaruga" (Super Lento para não cair) ---
    vectorstore = None
    batch_size = 1  # Processa APENAS 1 por vez

    progress_text = "Gerando Embeddings (Modo Lento: 1 a cada 5s)..."
    my_bar = st.progress(0, text=progress_text)

    total_splits = len(splits)

    for i in range(0, total_splits, batch_size):
        batch = splits[i:i+batch_size]

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
        except Exception as e:
            st.error(f"Erro no lote {i}: {e}")
            # Se der erro, espera mais tempo ainda antes de tentar o próximo
            time.sleep(10)
            continue

        # Atualiza barra
        percent_complete = min((i + 1) / total_splits, 1.0)
        my_bar.progress(percent_complete, text=f"Processando parte {i + 1} de {total_splits}...")

        # PAUSA LONGA: 5 segundos de espera
        time.sleep(5)

    my_bar.empty()
    return vectorstore

if uploaded_file and api_key:
    if "vectorstore" not in st.session_state:
        st.info("💡 Dica: Use PDFs pequenos (1 a 5 páginas) para testar.")

        if st.button("Iniciar Processamento (Pode demorar)"):
            with st.spinner("Iniciando..."):
                try:
                    st.session_state.vectorstore = process_pdf(uploaded_file)
                    st.success("PDF processado! Agora pode perguntar.")
                except Exception as e:
                    st.error(f"Erro fatal: {e}")

    if "vectorstore" in st.session_state:
        user_question = st.text_input("Pergunte algo sobre o documento:")

        if user_question:
            llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

            prompt = ChatPromptTemplate.from_template("""
                Responda com base no contexto:
                <context>{context}</context>
                Pergunta: {input}
            """)

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Gerando resposta..."):
                response = retrieval_chain.invoke({"input": user_question})
                st.write(response["answer"])

elif not api_key:
    st.warning("Insira a API Key para começar.")