import streamlit as st
import os
from dotenv import load_dotenv

# Importações do LangChain e Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# Carrega variáveis de ambiente (sua API Key)
load_dotenv()

# Configuração da Página
st.set_page_config(page_title="Chat com PDF - RAG", page_icon="🤖")
st.title("🤖 Chat com seus Documentos (RAG)")

# Sidebar para Upload e Configuração
with st.sidebar:
    st.header("Upload do Arquivo")
    uploaded_file = st.file_uploader("Envie seu PDF", type="pdf")
    api_key = st.text_input("Cole sua Google API Key", type="password")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# Função Principal
def process_pdf(uploaded_file):
    # Salva o arquivo temporariamente para o Loader ler
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 1. Carregamento (Load)
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # 2. Divisão (Split) - Quebra o texto em pedaços menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Embeddings & Vector Store - Transforma em números e guarda
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    return vectorstore

# Lógica da Aplicação
if uploaded_file and api_key:
    if "vectorstore" not in st.session_state:
        with st.spinner("Processando o PDF... Criando vetores..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.success("PDF processado! Pode perguntar.")

    # Interface de Chat
    user_question = st.text_input("Faça uma pergunta sobre o documento:")

    if user_question:
        # 4. Configuração do LLM (O Chef)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # 5. O Prompt (Instrução para o Chef)
        prompt = ChatPromptTemplate.from_template("""
            Responda a pergunta com base APENAS no contexto fornecido abaixo.
            Se a resposta não estiver no contexto, diga que não sabe.

            <context>
            {context}
            </context>

            Pergunta: {input}
        """)

        # 6. Criando a "Corrente" (Chain)
        # Cria a corrente que junta os documentos (stuff chain)
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Recuperador (retriever) é a interface de busca do banco vetorial
        retriever = st.session_state.vectorstore.as_retriever()

        # Corrente final de Recuperação
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Pensando..."):
            response = retrieval_chain.invoke({"input": user_question})
            st.write(response["answer"])

elif not api_key:
    st.warning("Por favor, insira sua API Key na barra lateral.")