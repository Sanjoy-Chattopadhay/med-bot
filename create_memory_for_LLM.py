from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"


def load_pdf_files(data):
    print(f"Loading PDF files from: {data}")

    # Check if directory exists
    if not os.path.exists(data):
        print(f"ERROR: Directory '{data}' does not exist!")
        return []

    # Check if directory has PDF files
    pdf_files = [f for f in os.listdir(data) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"ERROR: No PDF files found in '{data}' directory!")
        return []

    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    try:
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        print(f"ERROR loading documents: {str(e)}")
        return []


def create_chunks(extracted_data):
    print("Creating text chunks...")
    if not extracted_data:
        print("ERROR: No documents to chunk!")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks


def get_embedding_model():
    print("Loading embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("Embedding model loaded successfully")
        return embedding_model
    except Exception as e:
        print(f"ERROR loading embedding model: {str(e)}")
        return None


def create_and_save_vectorstore(text_chunks, embedding_model):
    print("Creating FAISS vector store...")

    if not text_chunks:
        print("ERROR: No text chunks to create vector store!")
        return False

    if embedding_model is None:
        print("ERROR: Embedding model not available!")
        return False

    try:
        db = FAISS.from_documents(text_chunks, embedding_model)

        # Create directory if it doesn't exist
        DB_FAISS_PATH = "vectorstore/db_faiss"
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

        db.save_local(DB_FAISS_PATH)
        print(f"Vector store saved successfully to: {DB_FAISS_PATH}")
        return True
    except Exception as e:
        print(f"ERROR creating/saving vector store: {str(e)}")
        return False


def main():
    print("Starting vector store creation process...")
    print("=" * 50)

    # Step 1: Load PDF files
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print("FAILED: Could not load any documents. Please check your data directory.")
        return

    # Step 2: Create chunks
    text_chunks = create_chunks(documents)
    if not text_chunks:
        print("FAILED: Could not create text chunks.")
        return

    # Step 3: Get embedding model
    embedding_model = get_embedding_model()
    if embedding_model is None:
        print("FAILED: Could not load embedding model.")
        return

    # Step 4: Create and save vector store
    success = create_and_save_vectorstore(text_chunks, embedding_model)

    if success:
        print("=" * 50)
        print("SUCCESS: Vector store created successfully!")
        print("You can now run the medical chatbot.")
    else:
        print("FAILED: Could not create vector store.")


if __name__ == "__main__":
    main()