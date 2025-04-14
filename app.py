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

# Database functions
def get_all_customers(db_name="customers.db"):
    """Fetch all customer records from the database"""
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
    """Insert a single customer record into the database"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO customers (name, phone_number, email_address, company)
        VALUES (?, ?, ?, ?)
    ''', (
        customer["name"],
        customer["phone_number"],
        customer["email_address"],
        customer["company"]
    ))
    conn.commit()
    conn.close()

def fuzzy_match_customer(extracted, customers, thresholds={
    "name": 80,
    "phone_number": 90,
    "email_address": 90,
    "company": 80
}):
    """Perform fuzzy matching to find if extracted data matches any customer"""
    for customer in customers:
        # Calculate similarity scores for each field (handle None values)
        name_score = fuzz.token_sort_ratio(extracted.get("name", "") or "", customer.get("name", "") or "") if extracted.get("name") and customer.get("name") else 0
        phone_score = fuzz.ratio(extracted.get("phone_number", "") or "", customer.get("phone_number", "") or "") if extracted.get("phone_number") and customer.get("phone_number") else 0
        email_score = fuzz.ratio(extracted.get("email_address", "") or "", customer.get("email_address", "") or "") if extracted.get("email_address") and customer.get("email_address") else 0
        company_score = fuzz.token_sort_ratio(extracted.get("company", "") or "", customer.get("company", "") or "") if extracted.get("company") and customer.get("company") else 0

        # Consider it a match if any key field (email or phone) is above threshold
        # or if name and company are both moderately similar
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

# Streamlit UI
st.title("Data matching usecase")
st.write("Upload an image to extract customer data, match it with the database, and add new records if needed.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Extract and Process Customer Data"):
        # Step 1: Extract raw_text using NanoNets OCR
        nanonets_url = "https://app.nanonets.com/api/v2/OCR/FullText"
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'image/jpeg')}

        with st.spinner("🔍 Extracting text from image..."):
            try:
                nanonets_response = requests.post(
                    nanonets_url,
                    auth=requests.auth.HTTPBasicAuth(nanonets_api_key, ''),
                    files=files
                )

                if nanonets_response.status_code == 200:
                    result = nanonets_response.json()
                    st.success("✅ OCR Text Extracted!")

                    # Extract raw_text from the response
                    raw_text = ""
                    try:
                        if "results" in result and result["results"]:
                            for res in result["results"]:
                                if isinstance(res, dict) and "0" in res and "page_data" in res["0"]:
                                    for page in res["0"]["page_data"]:
                                        if "raw_text" in page:
                                            raw_text = page["raw_text"]
                                            break
                                if raw_text:
                                    break

                        if not raw_text:
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

                            raw_text = find_raw_text(result) or ""

                    except Exception as e:
                        st.error(f"❌ Error parsing NanoNets response: {str(e)}")
                        st.json(result)
                        raw_text = ""

                    if not raw_text.strip():
                        st.warning("⚠️ No text found in the image.")
                    else:
                        st.write("**Raw Extracted Text:**")
                        st.text(raw_text)

                        # Step 2: Send raw_text to Groq for formatting
                        with st.spinner("🧠 Formatting text with Groq..."):
                            try:
                                chat_completion = groq_client.chat.completions.create(
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": (
                                                "You are a helpful assistant skilled at extracting structured information from raw text. "
                                                "Given the input text, extract the following fields: name, phone_number, email_address, and company. "
                                                "Return ONLY a valid JSON object with these fields. Do not include any explanatory text, comments, or extra characters outside the JSON. "
                                                "If a field is not found, set it to null. "
                                                "Use these rules: "
                                                "- Name: Look for a personal name, often with initials or titles (e.g., Vinothkumar.S). "
                                                "- Phone_number: Combine country code and number (e.g., +91 98844 24114 -> +919884424114). "
                                                "- Email_address: Identify standard email formats (e.g., info@mudhyoghr.com). "
                                                "- Company: Prefer multi-word names ending in Ltd, Inc, Corp, or similar (e.g., Mepzon HR Services India Pvt Ltd). "
                                                "Example: "
                                                '{"name": "John Doe", "phone_number": "+1234567890", "email_address": "john.doe@example.com", "company": "Acme Corp"}'
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": f"Extract name, phone_number, email_address, and company from the following text:\n{raw_text}"
                                        }
                                    ],
                                    model="llama-3.3-70b-versatile",
                                    temperature=0.5,
                                    max_completion_tokens=1024,
                                    top_p=1,
                                    stop=None,
                                    stream=False
                                )

                                formatted_result = chat_completion.choices[0].message.content
                                try:
                                    formatted_json = json.loads(formatted_result)
                                    st.success("✅ Formatted Output:")
                                    st.json(formatted_json)

                                    # Step 3: Fuzzy match with database
                                    with st.spinner("🔎 Checking database for matching customer..."):
                                        customers = get_all_customers()
                                        is_match, matched_customer, match_scores = fuzzy_match_customer(formatted_json, customers)

                                        if is_match:
                                            st.info("ℹ️ Customer already exists in the database!")
                                            st.write("**Matched Customer Details:**")
                                            st.json(matched_customer)
                                            st.write("**Match Scores:**")
                                            st.json(match_scores)
                                        else:
                                            st.warning("⚠️ New customer record found!")
                                            st.write("**Extracted Customer Data:**")
                                            st.json(formatted_json)

                                            # Prompt user to add the record
                                            if st.button("✅ Add this customer to the database"):
                                                try:
                                                    insert_customer("customers.db", formatted_json)
                                                    st.success("✅ New customer added to the database!")
                                                except Exception as e:
                                                    st.error(f"❌ Error adding customer to database: {str(e)}")

                                except json.JSONDecodeError:
                                    st.warning("⚠️ Groq response is not pure JSON. Attempting to extract JSON...")
                                    json_match = re.search(r'\{.*?\}', formatted_result, re.DOTALL)
                                    if json_match:
                                        try:
                                            formatted_json = json.loads(json_match.group(0))
                                            st.success("✅ Formatted Output (extracted from text):")
                                            st.json(formatted_json)

                                            # Step 3: Fuzzy match with database
                                            with st.spinner("🔎 Checking database for matching customer..."):
                                                customers = get_all_customers()
                                                is_match, matched_customer, match_scores = fuzzy_match_customer(formatted_json, customers)

                                                if is_match:
                                                    st.info("ℹ️ Customer already exists in the database!")
                                                    st.write("**Matched Customer Details:**")
                                                    st.json(matched_customer)
                                                    st.write("**Match Scores:**")
                                                    st.json(match_scores)
                                                else:
                                                    st.warning("⚠️ New customer record found!")
                                                    st.write("**Extracted Customer Data:**")
                                                    st.json(formatted_json)

                                                    # Prompt user to add the record
                                                    if st.button("✅ Add this customer to the database"):
                                                        try:
                                                            insert_customer("customers.db", formatted_json)
                                                            st.success("✅ New customer added to the database!")
                                                        except Exception as e:
                                                            st.error(f"❌ Error adding customer to database: {str(e)}")

                                        except json.JSONDecodeError:
                                            st.error("❌ Could not extract valid JSON from Groq response.")
                                            st.text(formatted_result)
                                    else:
                                        st.error("❌ No JSON found in Groq response.")
                                        st.text(formatted_result)

                            except Exception as e:
                                st.error(f"❌ Groq API error: {str(e)}")

                else:
                    st.error(f"❌ Nanonets Error {nanonets_response.status_code}: {nanonets_response.text}")
            except Exception as e:
                st.error(f"❌ Nanonets request error: {str(e)}")
