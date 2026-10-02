
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(document):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents=document)

    return chunks