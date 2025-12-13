from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunking import split_documents

pdf_path = "data/pdfs/Proyecto_ChatPDF_IA (1).pdf"

docs = load_pdf(pdf_path)
chunks = split_documents(docs)

print("Páginas cargadas:", len(docs))
print("Chunks generados:", len(chunks))

print("\nEjemplo de chunk:\n")
print(chunks[0].page_content[:500])
