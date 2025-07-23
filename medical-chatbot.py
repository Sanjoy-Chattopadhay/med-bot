import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="MedBot - AI Medical Assistant",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model,
                          allow_dangerous_deserialization=True)
    return db


def set_custom_template(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt


def load_llm(repo_id):
    llm = ChatGroq(
        model=repo_id,
        temperature=0.5,
        max_retries=2,
    )
    return llm


def main():
    st.title("🏥 MedBot - AI Medical Assistant")

    # Sidebar with three sections
    with st.sidebar:
        st.header("📖 How to Use")
        st.write("**Step 1:** Type your medical question in the chat")
        st.write("**Step 2:** Click Enter or use the chat input")
        st.write("**Step 3:** Get AI-powered answers from medical documents")
        st.write("**Step 4:** View sources to verify information")

        st.divider()

        # st.header("🏗️ How It's Made")
        st.header("**Architecture Overview:**")
        st.write("• **PDF Processing:** LangChain document loaders, Trained on The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
        st.write("• **Text Chunking:** Recursive character splitter")
        st.write("• **Embeddings:** HuggingFace MiniLM-L6-v2")
        st.write("• **Vector Store:** FAISS for similarity search")
        st.write("• **LLM:** Groq Llama 3.1 8B Instant")
        st.write("• **RAG Pipeline:** LangChain RetrievalQA")
        st.write("• **Frontend:** Streamlit")

        st.divider()

        st.header("🚀 Use Now")
        st.write("Ready to get medical insights? Start asking questions below!")
        st.info("💡 **Tip:** Be specific with your medical questions for better results")

        if st.button("🔄 New Conversation", type="primary"):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state without persistent history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Ask your medical questions here...")

    if prompt:
        # Display user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Custom prompt template
        custom_prompt_template = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide detailed and helpful medical information when available.

        context: {context}
        Question: {question}

        Start the answer directly with no small talk.
        """

        repo_id = "llama-3.1-8b-instant"

        try:
            with st.spinner("🧠 Analyzing your question..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(repo_id),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={
                        'prompt': set_custom_template(custom_prompt_template)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response['result']
                source_docs = response['source_documents']

                # Format source documents
                formatted_sources = ""
                if source_docs:
                    formatted_sources = "\n\n**📚 Sources:**\n"
                    for i, doc in enumerate(source_docs, 1):
                        metadata = doc.metadata
                        page = metadata.get('page_label', metadata.get('page', 'Unknown'))
                        source = metadata.get('source', 'Unknown').split("\\")[-1].split("/")[-1]
                        formatted_sources += f"**{i}.** {source} (Page {page})\n"

                result_to_show = result + formatted_sources

                # Display assistant response
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}\n\nPlease try again or check your setup."
            st.chat_message('assistant').markdown(error_msg)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': error_msg})

    # Footer disclaimer
    st.divider()
    st.markdown("""
    **⚠️ Medical Disclaimer:** This AI assistant provides information for educational purposes only. 
    Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment.
    """)


if __name__ == "__main__":
    main()