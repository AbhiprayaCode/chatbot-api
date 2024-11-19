from src.helper import load_pdf_file, load_csv_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
data_directory = 'C:/UNIVERSITY/SEMESTER 4/Artificial Intelligence/Project/Chatbot/End-To-End-Medical-Chatbot/Data'
extracted_data_pdf = load_pdf_file(data='./Data')
extracted_data2 = load_csv_file(data='./Data')
text_chunks = text_split(extracted_data2)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# Embed each chunk and upsert the embeddings into your Pinecone index.
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)