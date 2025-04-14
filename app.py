import os
import streamlit as st
import pandas as pd
import asyncio
import json
from dotenv import load_dotenv
from nanonets import NANONETSOCR
from groq import Groq
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from fuzzywuzzy import fuzz
from datetime import datetime
import random
import names

# Load environment variables
load_dotenv()

# Configuration
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FILE = "customers.csv"

# Initialize OCR and Groq client
model = NANONETSOCR()
model.set_token(NANONETS_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Database functions
def load_customers():
    return pd.read_csv(DATA_FILE)

def save_customer(name: str, phone: str, dob: str, address: str = None):
    df = load_customers()
    new_record = pd.DataFrame([{"name": name, "phone": phone, "dob": dob, "address": address or ""}])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return f"Customer {name} created with reference ID: CUST{df.index[-1] + 1:03d}"

# Document processing
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_entities_from_document(file_path: str):
    try:
        text = model.convert_to_string(file_path, formatting='lines')
        prompt = f"""
        Extract the following information from the text below and return it as a JSON object:
        - name: The full name of the person
        - phone: The phone number (if present)
        - dob: Date of birth (format: YYYY-MM-DD, if present)

        Text: {text}

        Return only the JSON object:
        {{
            "name": "extracted name or null",
            "phone": "extracted phone or null",
            "dob": "extracted dob or null"
        }}
        """
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return {"name": None, "phone": None, "dob": None}

# Fuzzy matching
def fuzzy_match_customer(extracted_info, threshold=80):
    df = load_customers()
    for _, row in df.iterrows():
        name_match = fuzz.ratio(extracted_info["name"].lower(), row["name"].lower()) if extracted_info["name"] else 0
        phone_match = extracted_info["phone"] == row["phone"] if extracted_info["phone"] and row["phone"] else False
        dob_match = extracted_info["dob"] == row["dob"] if extracted_info["dob"] and row["dob"] else False
        if name_match >= threshold or phone_match or dob_match:
            return row.to_dict()
    return None

# Custom tools
def search_customer_tool(input_str: str):
    try:
        extracted_info = json.loads(input_str) if input_str.startswith("{") else {"name": input_str}
        match = fuzzy_match_customer(extracted_info)
        if match:
            return f"Customer found: {match['name']}, Phone: {match['phone']}, DOB: {match['dob']}"
        return "No customer found."
    except Exception as e:
        return f"Error searching customer: {str(e)}"

def create_customer_tool(input_str: str):
    try:
        data = json.loads(input_str)
        result = save_customer(data["name"], data["phone"], data["dob"], data.get("address"))
        return result
    except Exception as e:
        return f"Error creating customer: {str(e)}"

# Initialize LangChain agent
def initialize_agent():
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    tools = [
        Tool(
            name="SearchCustomer",
            func=search_customer_tool,
            description="Search for a customer by name, phone, or DOB. Input should be a JSON string or name."
        ),
        Tool(
            name="CreateCustomer",
            func=create_customer_tool,
            description="Create a new customer. Input must be a JSON string with name, phone, and dob."
        ),
        Tool(
            name="CSVAnalysis",
            func=lambda x: create_csv_agent(ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"), DATA_FILE, allow_dangerous_code=True).run(x),
            description="Analyze customer data in CSV. Use for counting, filtering, or summarizing."
        )
    ]
    prompt = PromptTemplate.from_template(
        "Answer the user's query using the provided tools. If a document is uploaded, process it first. For general queries, use CSVAnalysis. For customer search or creation, use SearchCustomer or CreateCustomer. Current conversation: {history}\nUser: {input}\nAgent: {agent_scratchpad}"
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
def main():
    st.title("AI Customer MDM Chatbot")
    st.write("Upload a document (PDF, JPEG, or PNG) to check if a customer exists or chat to analyze customer data.")

    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "extracted_info" not in st.session_state:
        st.session_state.extracted_info = None

    # Document upload (Updated to accept PDF, JPEG, PNG)
    uploaded_file = st.file_uploader("Upload Customer Document (PDF, JPEG, or PNG)", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.extracted_info = extract_entities_from_document(file_path)
            os.remove(file_path)
            st.write("Extracted Info:", st.session_state.extracted_info)
            result = search_customer_tool(json.dumps(st.session_state.extracted_info))
            st.write(result)
            if "No customer found" in result:
                st.write("Would you like to create a new customer with these details?")
                if st.button("Create Customer"):
                    create_result = create_customer_tool(json.dumps(st.session_state.extracted_info))
                    st.success(create_result)
                    st.session_state.extracted_info = None

    # Chat interface
    st.subheader("Chat with the MDM Bot")
    user_input = st.text_input("Ask about customers or data analysis (e.g., 'How many customers do we have?')")
    if user_input:
        # Append to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process with agent
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke({
                "input": user_input,
                "history": history_str,
                "agent_scratchpad": ""
            })["output"]
        
        # Append response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.write(f"**You**: {msg['content']}")
            else:
                st.write(f"**Bot**: {msg['content']}")

if __name__ == "__main__":
    main()
