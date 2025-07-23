import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Step 1 - Setup LLM model 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
repo_id = "llama-3.1-8b-instant"


def load_llm(repo_id):
    print("Loading LLM model...")
    try:
        llm = ChatGroq(
            model=repo_id,
            temperature=0.5,
            max_retries=2,
        )
        print("LLM model loaded successfully")
        return llm
    except Exception as e:
        print(f"ERROR loading LLM: {str(e)}")
        return None


# Step 2 - Connect with FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

context: {context}
Question: {question}

Start the Answer directly no small talks. 
"""


def set_custom_template(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt


def load_vectorstore():
    print("Loading vector store...")

    # Check if vector store exists
    if not os.path.exists(DB_FAISS_PATH):
        print(f"ERROR: Vector store not found at {DB_FAISS_PATH}")
        print("Please run create_memory_for_LLM.py first to create the vector store.")
        return None

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully")
        return db
    except Exception as e:
        print(f"ERROR loading vector store: {str(e)}")
        return None


def create_qa_chain(llm, db):
    print("Creating QA chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_template(custom_prompt_template)}
        )
        print("QA chain created successfully")
        return qa_chain
    except Exception as e:
        print(f"ERROR creating QA chain: {str(e)}")
        return None


def main():
    print("Starting Medical Chatbot Test...")
    print("=" * 50)

    # Check API key
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        print("Please set your GROQ API key in the .env file.")
        return

    # Load components
    llm = load_llm(repo_id)
    if llm is None:
        return

    db = load_vectorstore()
    if db is None:
        return

    qa_chain = create_qa_chain(llm, db)
    if qa_chain is None:
        return

    print("=" * 50)
    print("Medical Chatbot is ready! (Type 'quit' to exit)")
    print("=" * 50)

    while True:
        user_query = input("\nEnter your medical query: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_query:
            print("Please enter a valid query.")
            continue

        print("\nProcessing your query...")
        try:
            response = qa_chain.invoke({'query': user_query})

            print("\n" + "=" * 50)
            print("ANSWER:")
            print(response['result'])

            print("\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(response['source_documents'], 1):
                metadata = doc.metadata
                page = metadata.get('page_label', metadata.get('page', 'Unknown'))
                source = metadata.get('source', 'Unknown').split("\\")[-1]
                print(f"{i}. {source} (Page {page})")

            print("=" * 50)

        except Exception as e:
            print(f"ERROR processing query: {str(e)}")


if __name__ == "__main__":
    main()