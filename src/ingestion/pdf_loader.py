from langchain_community.document_loaders import PyPDFLoader
def load_pdf(pdf_path: str):
    """
    Carga un PDF y devuelve una lista de Documentos LangChain
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
