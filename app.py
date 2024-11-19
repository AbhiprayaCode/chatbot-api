import streamlit as st
import requests
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from dotenv import load_dotenv
import uuid
from src.prompt import *
import os

load_dotenv()

# MongoDB configuration
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["medical_chatbot"]
collection = db["chat_history"]

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

# Embed each chunk and upsert the embeddings into Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="gemma-7b-it",
    temperature=1,
    max_tokens=1024,
    verbose=True,
)

# Define prompt with expected input variables 'context' and 'input'
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{history}\n{context}\nUser: {input}"),
    ]
)

# Initialize memory to store conversation context
memory = ConversationBufferMemory(
    memory_key="history",  # Key used for accessing conversation history in chain
    input_key="input",     # Specifies the main input variable
    return_messages=True   # Ensures history is returned as a list of messages
)

# Create an LLM chain that includes memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def get_session_id():
    # Check if a session ID already exists; otherwise, create a new one.
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())  # Generate a unique user/session ID
    return st.session_state["session_id"]

def save_to_mongo(user_input, bot_response, session_id):
    # Insert the conversation into MongoDB, include the session ID
    collection.insert_one({
        "session_id": session_id,
        "user_input": user_input,
        "bot_response": bot_response
    })
# # Function to fetch health data from API
# API_URL = 'https://data.go.id/api/v1/health_data_endpoint'  # Ganti dengan endpoint aktual
# API_KEY = 'YOUR_API_KEY'  # API key yang diperoleh dari registrasi (jika diperlukan)

# def fetch_health_data():
#     headers = {
#         'Authorization': f'Bearer {API_KEY}',  # Tambahkan jika API membutuhkan authorization
#         'Content-Type': 'application/json'
#     }
    
#     response = requests.get(API_URL, headers=headers)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         response.raise_for_status()



# Streamlit App
def main():
    st.set_page_config(page_title="Doctor AI Chatbot", layout="wide")

    # Sidebar with information
    st.sidebar.title("Doctor AI")
    st.sidebar.info("Doctor AI is a chatbot that provides information about diseases, health conditions, and medical information. Part of CareSense project created by President University Informatics Students. The project is focused on AI in Healthcare to provide advisement to patients who are unable to go to a doctor by giving a recomendations and consultation for commons diseases (except fatal diseases).")
    st.sidebar.title("CareSense")
    st.sidebar.info("CareSense is a startup company that is focused on providing AI in Healthcare. The company is founded by President University Informatics Students.")
    
    st.title("Doctor AI - Your Health Assistant")

    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display the conversation history
    st.subheader("Chat History")
    chat_history_container = st.container()

    # Display chat history from session state
    with chat_history_container:
        for chat in st.session_state["chat_history"]:
            st.chat_message("user").write(f"{chat['user_input']}")
            st.chat_message("assistant").write(f"{chat['bot_response']}")

    # Form for user input
    with st.form(key='user_input_form', clear_on_submit=True):
        user_input = st.text_input("Type your message here:")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        # Display user message
        with chat_history_container:
            st.chat_message("user").write(f"{user_input}")

        # Update memory with user input to maintain history
        memory.chat_memory.add_user_message(user_input)

        # Get documents related to the user query using retriever
        related_docs = retriever.get_relevant_documents(user_input)
        
        # Format related documents into a single string for better context
        context = "\n".join([doc.page_content for doc in related_docs])

        # Invoke the conversation chain with the user message, history, and context from related documents
        response = conversation_chain({"input": user_input, "context": context})
        bot_response = response["text"]

        # Update memory with bot response to maintain history
        memory.chat_memory.add_ai_message(bot_response)

        # Display bot response
        with chat_history_container:
            st.chat_message("assistant").write(f"{bot_response}")

        # Save the conversation to session state
        st.session_state["chat_history"].append({"user_input": user_input, "bot_response": bot_response})

        # Get the existing session ID or create a new one
        session_id = get_session_id()

        # Save the conversation to MongoDB
        save_to_mongo(user_input, bot_response, session_id)

if __name__ == "__main__":
    main()
