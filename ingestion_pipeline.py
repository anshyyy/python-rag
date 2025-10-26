import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


def load_documents(directory_path = "docs"):
    '''Load documents from a specified directory.'''
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")

    
    loader = DirectoryLoader(
        directory_path, 
        glob="*.txt", 
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in the directory: {directory_path}")
    
    for i ,doc in  enumerate(documents[:2]):
        print(f"Loaded document {i+1}: {doc.metadata.get('source', 'Unknown Source')} with {len(doc.page_content)} characters.")
        print(f"content metadata: {doc.metadata}")

    return documents
 

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    '''Split documents into smaller chunks for processing.'''
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)

    if split_docs:
        print(f"First chunk content preview: {split_docs[0].page_content[:200]}...")    
        print(f"First chunk metadata: {split_docs[0].metadata}")

    
    print(f"Total chunks created: {len(split_docs)}")
    return split_docs


def create_vector_store(documents, persist_directory="db/vector_store"):
    '''Create and persist a vector store from the documents.'''

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small" 
    )

    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created at: {persist_directory}")
    return vector_store

def main():
    print("Starting ingestion pipeline...")
    # Load documents from the specified directory
    documents = load_documents()

    # chuking to handle large files efficiently
    split_docs = split_documents(documents)


    # embedding and storing in vector database
    vector_store = create_vector_store(split_docs)


if __name__ == "__main__":
    main()