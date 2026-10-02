from langchain_community.document_loaders import TextLoader, PyPDFLoader

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path=file_path)
    document = loader.load()

    return document

def text_loader(file_path):
    loader = TextLoader(file_path=file_path)
    document = loader.load()

    return document