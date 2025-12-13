# ===============================
# Baseline RAG (Single-Hop)
# ===============================

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import OllamaLLM


# 1. Cargar PDF
pdf_path = "data/pdfs/Proyecto_ChatPDF_IA (1).pdf"
docs = load_pdf(pdf_path)

# 2. Chunking
chunks = split_documents(docs)
print(f"Páginas cargadas: {len(docs)}")
print(f"Chunks generados: {len(chunks)}")

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6. Prompt SIMPLE (clave para que no se cuelgue)
prompt = PromptTemplate.from_template(
    """
Usa SOLO el contexto para responder la pregunta.
Si no sabes la respuesta, di "No se encontró en el documento".

Contexto:
{context}

Pregunta:
{question}

Respuesta breve:
"""
)

# 7. LLM (Ollama)
llm = OllamaLLM(
    model="llama3",
    temperature=0
)

# 8. Cadena RAG
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# 9. Pregunta de prueba
query = "¿Quienes son los integrantes?"
print("\nPregunta:", query)

answer = rag_chain.invoke(query)

print("\nRespuesta:")
print(answer)
