# RAG Document Ingestion Script for ESI Scholarly Instructor
"""
This script handles the ingestion of documents into a ChromaDB vector store
for Retrieval Augmented Generation (RAG) by the ESI Scholarly Instructor agent.

It performs the following steps:
1. Loads documents (PDF, TXT, MD) from a specified source directory.
2. Splits the loaded documents into smaller, manageable chunks.
3. Initializes Google Generative AI Embeddings.
4. Creates or updates a ChromaDB collection with the embedded document chunks.

To use this script:
1. Ensure GOOGLE_API_KEY is set in your .env file.
2. Create a directory (default: './source_documents').
3. Place the documents you want to ingest into this directory.
4. Run the script from your terminal: `python ingest.py`
"""

import os
import glob # For checking if files exist before DirectoryLoader tries to load them
from dotenv import load_dotenv
from typing import List

# Langchain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# Load from .env or use defaults
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_esi")
"""Directory where the ChromaDB vector store will be persisted."""

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "esi_rag_collection")
"""Name of the collection within ChromaDB."""

SOURCE_DOCUMENTS_PATH = "./source_documents"
"""Directory where source documents for RAG ingestion should be placed."""

# Supported extensions for ingestion and their corresponding Langchain loaders
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}

# Parameters for document chunking
CHUNK_SIZE = 1500
"""Target size for text chunks (in characters)."""
CHUNK_OVERLAP = 200
"""Number of characters to overlap between chunks."""


# --- Document Loading ---
def load_source_documents(source_dir: str) -> List[Document]:
    """
    Loads all supported documents (PDF, TXT, MD) from the specified source directory.
    Uses DirectoryLoader for efficient loading of multiple files.

    Args:
        source_dir (str): The path to the directory containing source documents.

    Returns:
        List[Document]: A list of Langchain Document objects loaded from the files.
                        Returns an empty list if the directory doesn't exist or no supported files are found.
    """
    docs: List[Document] = []
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found. Please create it and add documents.")
        return docs

    for ext, loader_class in LOADER_MAPPING.items():
        # Construct glob pattern for the current extension
        # Using recursive glob `**/*{ext}` to find files in subdirectories as well
        file_pattern = os.path.join(source_dir, f"**/*{ext}")

        # Check if any files match the pattern before attempting to load
        # This prevents DirectoryLoader from raising an error if no files of a type are found.
        if not glob.glob(file_pattern, recursive=True):
            print(f"No files found for extension '{ext}' in '{source_dir}'.")
            continue

        print(f"Loading files with extension '{ext}' from '{source_dir}'...")
        try:
            # Specific loader arguments, e.g., encoding for text files
            loader_kwargs = {'encoding': 'utf-8'} if ext == ".txt" else {}

            directory_loader = DirectoryLoader(
                source_dir,
                glob=f"**/*{ext}", # Ensure this matches the glob used for checking
                loader_cls=loader_class,
                loader_kwargs=loader_kwargs,
                show_progress=True,      # Display a progress bar during loading
                use_multithreading=True, # Speed up loading for many files
                silent_errors=True       # Log errors for individual files but continue processing others
            )
            loaded_docs_for_ext = directory_loader.load()

            if loaded_docs_for_ext:
                docs.extend(loaded_docs_for_ext)
                print(f"Loaded {len(loaded_docs_for_ext)} document(s) of type '{ext}'.")
            else:
                # This case might be hit if files were found but were empty or unreadable by the loader
                print(f"No documents of type '{ext}' were successfully loaded (files might be empty or unreadable).")
        except Exception as e:
            # Catch any other unexpected errors during DirectoryLoader instantiation or loading
            print(f"Error loading files with extension '{ext}': {e}")
            # Continue to the next extension type

    # Add a 'filename' field to metadata for easier identification
    for doc in docs:
        if 'source' in doc.metadata:
            doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
        else:
            # This case should ideally not be hit if DirectoryLoader is used, as it populates 'source'.
            doc.metadata['filename'] = 'unknown_source_file'

    return docs

# --- Text Splitting ---
def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a list of Langchain Documents into smaller chunks suitable for embedding.

    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        List[Document]: A list of new Document objects, where each represents a chunk.
    """
    if not documents:
        print("No documents provided for splitting.")
        return []

    print(f"Splitting {len(documents)} document(s) into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,      # Use character count for length
        add_start_index=True,     # Include the start index of the chunk in metadata
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(splits)} text chunks.")
    return splits

# --- Main Ingestion Logic ---
def main_ingest():
    """
    Main function to orchestrate the document ingestion process.
    Loads, splits, embeds, and stores documents in ChromaDB.
    """
    print("--- Starting ESI Scholarly Instructor RAG Ingestion Process ---")

    # Step 0: Ensure source_documents directory exists
    if not os.path.exists(SOURCE_DOCUMENTS_PATH):
        os.makedirs(SOURCE_DOCUMENTS_PATH)
        print(f"Created source documents directory: '{SOURCE_DOCUMENTS_PATH}'")
        print(f"Please add your PDF, TXT, or MD documents for RAG to this directory and then re-run the ingestion script.")
        return # Exit if directory was just created, user needs to add files

    # Step 1: Initialize Embeddings
    print("\nInitializing embeddings model...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env file or environment variables. Cannot proceed with embedding.")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        print("Google Generative AI Embeddings initialized successfully.")
    except Exception as e:
        print(f"ERROR: Could not initialize Google Generative AI Embeddings: {e}")
        return

    # Step 2: Load Documents
    print(f"\nStep 1: Loading documents from '{SOURCE_DOCUMENTS_PATH}'...")
    documents = load_source_documents(SOURCE_DOCUMENTS_PATH)
    if not documents:
        print("No documents found to ingest. Please ensure supported files (PDF, TXT, MD) are in the "
              f"'{SOURCE_DOCUMENTS_PATH}' directory and its subdirectories.")
        return
    print(f"Successfully loaded a total of {len(documents)} source document(s).")

    # Step 3: Split Documents into Chunks
    print("\nStep 2: Splitting documents into manageable chunks...")
    doc_splits = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not doc_splits:
        print("No text chunks were generated from the documents. This might happen if documents are empty or very small.")
        return

    # Step 4: Initialize ChromaDB Vectorstore and Add Document Chunks
    print(f"\nStep 3: Preparing to add document chunks to ChromaDB vectorstore...")
    print(f"   - Persist directory: '{CHROMA_PERSIST_DIR}'")
    print(f"   - Collection name: '{CHROMA_COLLECTION_NAME}'")

    # Ensure the ChromaDB persistence directory exists
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        print(f"Created ChromaDB persist directory: '{CHROMA_PERSIST_DIR}'")

    try:
        # Chroma.from_documents will create the collection if it doesn't exist,
        # or add to it if it does (assuming embedding function and collection name match).
        print(f"Adding {len(doc_splits)} document chunks to ChromaDB. This may take some time...")
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        vectorstore.persist() # Ensure data is written to disk
        print(f"\nSuccessfully added/updated {len(doc_splits)} document chunks in ChromaDB.")
        print("Vectorstore persisted.")
    except Exception as e:
        print(f"ERROR: An error occurred during ChromaDB initialization or document addition: {e}")
        print("Please ensure ChromaDB is installed correctly (e.g., `pip install chromadb`).")
        return

    print("\n--- RAG Ingestion Process Completed Successfully! ---")
    print(f"Your knowledge base in '{CHROMA_PERSIST_DIR}' (collection: '{CHROMA_COLLECTION_NAME}') is ready for the ESI agent.")


if __name__ == "__main__":
    main_ingest()
