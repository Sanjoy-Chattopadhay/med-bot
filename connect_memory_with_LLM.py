import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
#step 1 - Setup LLM model 

GROQ_API_KEY =os.environ.get("GROQ_API_KEY")
repo_id = "llama-3.1-8b-instant"

def load_llm(repo_id):
    llm = ChatGroq(
        model = repo_id,
        temperature=0.5,
        max_retries=2,
    )
    return llm

#step 2 - Connect with FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """
Use the peices of information provided in the context to answer user's question.
If you dont know the answer, just say that you dint know, dont try to make up an answer.

context : {context}
Question : {question}

Start the Answer directly no small talks. 
"""
def set_custom_template(custom_prompt_template):
    prompt = PromptTemplate(
        template = custom_prompt_template,
        input_variables=['context','question']
    )
    return prompt

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH ,embedding_model,allow_dangerous_deserialization=True)

#step 3 - Create Chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(repo_id),
    chain_type="stuff",
    retriever = db.as_retriever(search_kwargs = {'k':3}), #search_kwargs is the number of chunks the llm will search for a question .... in this case only top 3 chunks that has the information will be searched
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_template(custom_prompt_template)}
)

user_query = input("write query here")

response = qa_chain.invoke({'query':user_query})

print("RESULT: ", response['result'])
print("SOURCE_DOCUMENTS: ", response['source_documents'])