import os
from langchain_pinecone import PineconeVectorStore
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import get_embedding_model
from src.prompt import *
from flask import Flask, render_template, Response, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

embedding_model = get_embedding_model()

index_name = "medical-chatbot"

# get existing pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k" : 3})

# llm = OllamaLLM(model="llama3.1")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]   
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# healthcheck
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    # user_input = request.form["msg"]
    user_input = request.args.get("user_input")
    print("input : ", input)
    response = rag_chain.invoke({"input" : user_input})
    print("response : ", response["answer"])
    return str(response["answer"])
    
    vector_store = PineconeVectorStore.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), embedding_model)
        
if __name__ =="__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)