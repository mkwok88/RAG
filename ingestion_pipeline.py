# ingestion_pipeline.py is used to chunk all the source documents and embed them to the vector databse
# this files will have a couple of dependencies 

import os 
# both these classes help with reading text files, ppt or txt
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# this class to chunk 
from langchain_text_splitters import CharacterTextSplitter
# to embed the text into a vector database
from langchain_openai import OpenAIEmbeddings
# vector DB we're using is Chroma, we can host this locally
from langchain_chroma import Chroma
# if we have a .env file, we can load the environment variables from it
from dotenv import load_dotenv
# to accept UTF-8 format of text documents
from functools import partial

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        # only look for txt files
        glob="*.txt",
        # right now only text files, but pdf loaders and ppt loaders also exists for those caases
         loader_cls=partial(TextLoader, encoding="utf-8")
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   # Show that it's working
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # if chunks:
    
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)
        
    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        # where we store our vector database, local for chroma
        persist_directory=persist_directory, 
        # cosine similarity used to find the most similar chunks to the query
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main function")

    #1 Load the files
    documents = load_documents(docs_path="docs")

    #2 Chunking the files
    chunks = split_documents(documents)

    #3 Send the chunks to the embedding model and convert and store in the vector database
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()

