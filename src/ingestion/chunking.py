from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=800, overlap=150):
    """
    Divide los documentos en fragmentos (chunks)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks
