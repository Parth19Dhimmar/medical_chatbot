from src.helper import load_pdf_file, text_split, get_embedding_model

import os
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

extracted_data = load_pdf_file(data="data/")
text_chunks=text_split(extracted_data)
embedding_model=get_embedding_model()

# pinecone initialization
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create an index
index_name="medical-chatbot"

index_config = pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
)

# create vector store
vector_store = PineconeVectorStore.from_documents(
    embedding=get_embedding_model(),
    index_name="medical-chatbot",
    documents=text_chunks)
