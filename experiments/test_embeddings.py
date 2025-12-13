from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# 4. Vector Store FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Prueba de búsqueda
query = "¿Cuál es el objetivo del proyecto?"
results = vectorstore.similarity_search(query, k=3)

print("\nResultados de prueba:\n")
for i, r in enumerate(results):
    print(f"--- Resultado {i+1} ---")
    print(r.page_content[:300])
    print()
