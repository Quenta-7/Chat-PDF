import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuración Global
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
current_retriever = None
current_chain = None
processed_docs_store = {} 

def process_document(file_path):
    global current_retriever, processed_docs_store
    
    # 1. Cargar
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Chunking (Aumentamos un poco el overlap para no cortar frases clave)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300, 
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Indexación Híbrida
    vectorstore = FAISS.from_documents(splits, embeddings)
    # CAMBIO 1: Aumentamos k a 7 para recuperar más pistas
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 7 # También aquí
    
    current_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    
    processed_docs_store = {i: doc for i, doc in enumerate(splits)}
    
    return len(splits), len(docs)

def get_answer(query):
    global current_retriever
    
    if not current_retriever:
        return "Por favor, sube un documento primero.", []

    retrieved_docs = current_retriever.invoke(query)
    context_text = "\n\n".join([f"[FRAGMENTO PÁG {d.metadata.get('page', 0)+1}]: {d.page_content}" for d in retrieved_docs])

    # Prompt reforzado para obligar al modelo a conectar puntos
    template = """Eres un asistente académico experto de la UNSAAC especializado en análisis forense de documentos.
    
    INSTRUCCIONES CLAVE:
    1. Tu objetivo principal es el RAZONAMIENTO MULTI-HOP: Debes conectar hechos dispersos.
    2. Ejemplo: Si el Texto A dice "El Proyecto X usa el componente Y" y el Texto B dice "El componente Y falla a 50°C", DEBES CONCLUIR que "El Proyecto X falla a 50°C".
    3. Cita siempre las páginas de donde sacas cada dato.
    
    CONTEXTO RECUPERADO:
    {context}
    
    PREGUNTA DEL USUARIO: {question}
    
    RESPUESTA RAZONADA:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # CAMBIO 2: Usamos Llama 3.3 70B (El modelo más potente y actual de Groq)
    # Si este diera error, usa "llama-3.1-70b-versatile"
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response_text = chain.invoke(query)
    
    sources = []
    seen_pages = set()
    for doc in retrieved_docs:
        page = doc.metadata.get('page', 0) + 1
        content_preview = doc.page_content[:150].replace("\n", " ") + "..."
        if page not in seen_pages:
            sources.append({"page": page, "text": content_preview})
            seen_pages.add(page)
            
    return response_text, sources