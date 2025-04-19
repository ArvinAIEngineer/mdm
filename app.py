import streamlit as st
import requests
import os
from dotenv import load_dotenv
from groq import Groq
import json
import re
import sqlite3
from fuzzywuzzy import fuzz

# Load API keys from .env file
load_dotenv()
nanonets_api_key = os.getenv("NANONETS_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Setup session state
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = "init"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "add_result" not in st.session_state:
    st.session_state.add_result = None

# Database functions
def get_all_customers(db_name="customers.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT name, phone_number, email_address, company FROM customers")
    customers = [
        {"name": row[0], "phone_number": row[1], "email_address": row[2], "company": row[3]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return customers

def insert_customer(db_name, customer):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO customers (name, phone_number, email_address, company)
        VALUES (?, ?, ?, ?)
    ''', (
        customer["name"], customer["phone_number"], customer["email_address"], customer["company"]
    ))
    conn.commit()
    conn.close()

def fuzzy_match_customer(extracted, customers, thresholds={
    "name": 80,
    "phone_number": 90,
    "email_address": 90,
    "company": 80
}):
    for customer in customers:
        name_score = fuzz.token_sort_ratio(extracted.get("name", "") or "", customer.get("name", "") or "")
        phone_score = fuzz.ratio(extracted.get("phone_number", "") or "", customer.get("phone_number", "") or "")
        email_score = fuzz.ratio(extracted.get("email_address", "") or "", customer.get("email_address", "") or "")
        company_score = fuzz.token_sort_ratio(extracted.get("company", "") or "", customer.get("company", "") or "")

        if (email_score >= thresholds["email_address"] or
            phone_score >= thresholds["phone_number"] or
            (name_score >= thresholds["name"] and company_score >= thresholds["company"])):
            return True, customer, {
                "name_score": name_score,
                "phone_score": phone_score,
                "email_score": email_score,
                "company_score": company_score
            }
    return False, None, None

def extract_text_from_image(uploaded_file):
    nanonets_url = "https://app.nanonets.com/api/v2/OCR/FullText"
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'image/jpeg')}
    response = requests.post(
        nanonets_url,
        auth=requests.auth.HTTPBasicAuth(nanonets_api_key, ''),
        files=files
    )
    if response.status_code != 200:
        return ""
    result = response.json()
    def find_raw_text(obj):
        if isinstance(obj, dict):
            if "raw_text" in obj:
                return obj["raw_text"]
            for value in obj.values():
                result = find_raw_text(value)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_raw_text(item)
                if result:
                    return result
        return None
    return find_raw_text(result) or ""

def extract_entities_with_groq(raw_text):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant skilled at extracting structured information from raw text. "
                    "Extract name, phone_number, email_address, and company. Return JSON only."
                )},
                {"role": "user", "content": raw_text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024
        )
        result = chat_completion.choices[0].message.content
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            return json.loads(json_match.group(0)) if json_match else None
    except Exception as e:
        return None

# UI Title
st.title("\U0001F4C7 Customer Info Chatbot")

# Form submission
if st.session_state.add_result == "success":
    st.success("✅ Customer added to database.")
    st.session_state.add_result = None
    st.session_state.extracted_data = None
elif st.session_state.add_result == "error":
    st.error("❌ Failed to add customer.")
    st.session_state.add_result = None

# Chatbot flow
if st.session_state.chat_stage == "init":
    user_input = st.chat_input("Type 'hi' to begin")
    if user_input:
        if user_input.lower().strip() == "hi":
            st.session_state.chat_stage = "waiting_for_choice"
            st.experimental_rerun()
        else:
            with st.chat_message("bot"):
                st.markdown("❓ Please type 'hi' to get started.")

elif st.session_state.chat_stage == "waiting_for_choice":
    with st.chat_message("bot"):
        st.markdown("Hi! Do you want to upload a document or enter customer details manually?")
    user_input = st.chat_input("Type 'upload' or 'enter'")
    if user_input:
        if "upload" in user_input.lower():
            st.session_state.chat_stage = "show_upload"
            st.experimental_rerun()
        elif "enter" in user_input.lower():
            st.session_state.chat_stage = "manual_entry"
            st.experimental_rerun()

elif st.session_state.chat_stage == "show_upload":
    uploaded_file = st.file_uploader("Upload a document", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.chat_stage = "process_upload"
        st.experimental_rerun()

elif st.session_state.chat_stage == "process_upload":
    uploaded_file = st.session_state.uploaded_file
    with st.chat_message("bot"):
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
        st.markdown("🔍 Extracting text from image...")
    raw_text = extract_text_from_image(uploaded_file)
    safe_text = raw_text.encode('utf-8', 'replace').decode('utf-8')
    with st.chat_message("bot"):
        st.markdown("🧠 Running LLM to format extracted text...")
        st.text(safe_text)
    extracted_data = extract_entities_with_groq(safe_text)
    st.session_state.extracted_data = extracted_data
    st.session_state.chat_stage = "show_result"
    st.experimental_rerun()

elif st.session_state.chat_stage == "manual_entry":
    with st.chat_message("bot"):
        st.markdown("Please enter the customer details:")
    user_input = st.chat_input("Type details here")
    if user_input:
        extracted_data = extract_entities_with_groq(user_input)
        st.session_state.extracted_data = extracted_data
        st.session_state.chat_stage = "show_result"
        st.experimental_rerun()

elif st.session_state.chat_stage == "show_result":
    extracted_data = st.session_state.extracted_data
    if extracted_data:
        with st.chat_message("bot"):
            st.markdown("✅ Extracted customer data:")
            for key, value in extracted_data.items():
                st.write(f"**{key.capitalize()}**: {value}")
        customers = get_all_customers()
        is_match, matched, scores = fuzzy_match_customer(extracted_data, customers)
        if is_match:
            st.info("✅ Customer already exists:")
            for key, value in matched.items():
                st.write(f"**{key.capitalize()}**: {value}")
            st.write("**Match Scores:**")
            for key, value in scores.items():
                st.write(f"{key}: {value}")
        else:
            st.warning("🆕 This is a new customer.")
            with st.form("add_customer_form"):
                name = st.text_input("Name", value=extracted_data.get("name", ""))
                phone = st.text_input("Phone Number", value=extracted_data.get("phone_number", ""))
                email = st.text_input("Email Address", value=extracted_data.get("email_address", ""))
                company = st.text_input("Company", value=extracted_data.get("company", ""))
                if st.form_submit_button("✅ Add to database"):
                    try:
                        insert_customer("customers.db", {
                            "name": name,
                            "phone_number": phone,
                            "email_address": email,
                            "company": company
                        })
                        st.session_state.add_result = "success"
                    except:
                        st.session_state.add_result = "error"
                    st.session_state.chat_stage = "waiting_for_choice"
                    st.experimental_rerun()
    else:
        st.error("❌ Could not extract structured data.")
        st.session_state.chat_stage = "waiting_for_choice"
