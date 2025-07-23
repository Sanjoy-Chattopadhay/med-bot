import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


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
    st.title("Med Bot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your queries")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        custom_prompt_template = """
        Use the peices of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dint know, dont try to make up an answer.

        context : {context}
        Question : {question}

        Start the Answer directly no small talks. 
        """
        repo_id = "llama-3.1-8b-instant"
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
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
            for i, doc in enumerate(source_docs, 1):
                metadata = doc.metadata
                page = metadata.get('page_label', metadata.get('page', 'Unknown'))
                source = metadata.get('source', 'Unknown').split("\\")[-1]
                formatted_sources += f"**{i}. {source} (Page {page})**\n"

            result_to_show = result + '\n\n**Sources:**\n' + formatted_sources
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': result_to_show})
        except Exception as e:
            st.error(f"Error: {str(e)}")



if __name__ == "__main__":
    main()
