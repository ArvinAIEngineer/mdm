import os
import streamlit as st
import asyncio
import asyncpg
import sqlite3
from dotenv import load_dotenv
from nanonets import NANONETSOCR
from groq import Groq
import json
from typing import Dict, Optional, List, Tuple
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

# Configuration
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "customers.db"  # SQLite database

# Initialize OCR and LLM
model = NANONETSOCR()
model.set_token(NANONETS_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Database functions for SQLite
def init_database():
    """Initialize the SQLite database with necessary tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create customers table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT,
        address TEXT,
        verification_status TEXT DEFAULT 'pending'
    )
    ''')
    
    conn.commit()
    conn.close()

def get_all_customers() -> List[Dict]:
    """Get all customers from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Explicitly include the id column in the query
    cursor.execute('SELECT id, name, phone, address, verification_status FROM customers')
    customers = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return customers

def add_customer(name: str, phone: str, address: str) -> int:
    """Add a new customer to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO customers (name, phone, address) VALUES (?, ?, ?)',
        (name, phone, address)
    )
    
    customer_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return customer_id

def update_verification_status(customer_id: int, status: str):
    """Update verification status for a customer"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        'UPDATE customers SET verification_status = ? WHERE id = ?',
        (status, customer_id)
    )
    
    conn.commit()
    conn.close()

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return the path"""
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_extraction_prompt(doc_type: str, text: str) -> str:
    """Generate appropriate prompt based on document type"""
    base_prompt = """
    Extract the following information from the text below and return it as a JSON object:
    - name: The full name of the person
    - phone: The phone number (if present)
    - address: The complete address

    Text: {text}

    Important: Look carefully through the entire text for these details. They might appear anywhere in the document.
    The name might be preceded by terms like "Name:", "Customer Name:", etc.
    The address might be preceded by "Address:", "Residence:", "Location:", etc.
    Phone numbers might be in various formats including +91 prefix or 10 digits.

    Return only the JSON object in this format:
    {{
        "name": "extracted name or null",
        "phone": "extracted phone or null",
        "address": "extracted address or null"
    }}
    """
    
    if doc_type == "id":
        base_prompt += "\nNote: This is an ID document. Look for officially stated name and address."
    elif doc_type == "bank":
        base_prompt += "\nNote: This is a bank statement. Look for account holder details and registered address."
    
    return base_prompt.format(text=text)

def extract_entities_using_groq(text: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Extract entities from text using Groq's language model"""
    prompt = get_extraction_prompt(doc_type, text)

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=500,
        )
        
        # Log the raw text and extracted information for debugging
        print(f"\nProcessing {doc_type.upper()} document")
        print("Raw OCR text:", text[:500] + "..." if len(text) > 500 else text)
        print("Groq response:", response.choices[0].message.content)
        
        # Parse the response into a dictionary
        extracted_info = json.loads(response.choices[0].message.content)
        return {
            'name': extracted_info.get('name'),
            'phone': extracted_info.get('phone'),
            'address': extracted_info.get('address')
        }
    except Exception as e:
        print(f"Error processing {doc_type} document:", str(e))
        return {'name': None, 'phone': None, 'address': None}

def process_document(file_path: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Process document using OCR and Groq for entity extraction"""
    try:
        # Get OCR text
        text = model.convert_to_string(file_path, formatting='lines')
        
        # Extract entities using Groq
        return extract_entities_using_groq(text, doc_type)
    except Exception as e:
        print(f"Error in document processing: {str(e)}")
        return {'name': None, 'phone': None, 'address': None}

def compare_with_fuzzy_match(str1: str, str2: str, threshold: int = 80) -> bool:
    """Compare two strings using fuzzy matching"""
    if not str1 or not str2:
        return False
    
    # Clean the strings
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    
    # Get the ratio
    ratio = fuzz.ratio(str1, str2)
    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)
    
    # Use the higher of the two scores
    similarity = max(ratio, token_sort_ratio)
    
    return similarity >= threshold, similarity

def compare_phone_numbers(phone1: str, phone2: str) -> Tuple[bool, int]:
    """Compare phone numbers by removing all non-digits"""
    if not phone1 or not phone2:
        return False, 0
    
    # Remove all non-digits
    phone1 = ''.join(filter(str.isdigit, phone1))
    phone2 = ''.join(filter(str.isdigit, phone2))
    
    # If both are empty after cleaning, return False
    if not phone1 or not phone2:
        return False, 0
    
    # Compare the last 10 digits if longer
    phone1 = phone1[-10:] if len(phone1) >= 10 else phone1
    phone2 = phone2[-10:] if len(phone2) >= 10 else phone2
    
    exact_match = phone1 == phone2
    similarity = 100 if exact_match else 0
    
    return exact_match, similarity

def find_matching_customers(extracted_info: Dict, customers: List[Dict]) -> List[Dict]:
    """Find matching customers in the database using fuzzy matching"""
    matching_results = []
    
    for customer in customers:
        matches = []
        mismatches = []
        match_details = {}
        total_score = 0
        fields_compared = 0
        
        # Compare name with fuzzy matching
        if extracted_info['name'] and customer['name']:
            name_match, name_score = compare_with_fuzzy_match(extracted_info['name'], customer['name'])
            total_score += name_score
            fields_compared += 1
            
            if name_match:
                matches.append('name')
                match_details['name'] = f"Name matched with {name_score}% confidence"
            else:
                mismatches.append('name')
                match_details['name'] = f"Names differ ({name_score}% similarity)"
        
        # Compare phone with exact matching
        if extracted_info['phone'] and customer['phone']:
            phone_match, phone_score = compare_phone_numbers(extracted_info['phone'], customer['phone'])
            total_score += phone_score
            fields_compared += 1
            
            if phone_match:
                matches.append('phone')
                match_details['phone'] = "Phone numbers match exactly"
            else:
                mismatches.append('phone')
                match_details['phone'] = "Phone numbers differ"
        
        # Compare address with fuzzy matching
        if extracted_info['address'] and customer['address']:
            address_match, address_score = compare_with_fuzzy_match(extracted_info['address'], customer['address'])
            total_score += address_score
            fields_compared += 1
            
            if address_match:
                matches.append('address')
                match_details['address'] = f"Addresses match with {address_score}% similarity"
            else:
                mismatches.append('address')
                match_details['address'] = f"Addresses differ ({address_score}% similarity)"
        
        # Calculate average matching score
        avg_score = total_score / fields_compared if fields_compared > 0 else 0
        
        # Add to matching results if at least one field matches
        if matches:
            matching_results.append({
                'customer': customer,
                'matches': matches,
                'mismatches': mismatches,
                'match_details': match_details,
                'avg_score': avg_score,
                'num_matches': len(matches)
            })
    
    # Sort by number of matches and then by average score
    matching_results.sort(key=lambda x: (x['num_matches'], x['avg_score']), reverse=True)
    
    return matching_results

def main():
    # Initialize the database
    init_database()
    
    st.title("Customer Document Verification")
    
    # Upload document for verification
    uploaded_file = st.file_uploader("Upload Customer Document (PDF)", type="pdf")
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save and process file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Process document (we'll use ID type for generic extraction)
                    extracted_info = process_document(file_path, "id")
                    
                    # Display extracted information
                    st.subheader("Extracted Information")
                    st.write(f"Name: {extracted_info['name']}")
                    st.write(f"Phone: {extracted_info['phone']}")
                    st.write(f"Address: {extracted_info['address']}")
                    
                    # Get all customers from database
                    customers = get_all_customers()
                    
                    # Find matching customers
                    matching_results = find_matching_customers(extracted_info, customers)
                    
                    # Display matching results
                    st.subheader("Database Matching Results")
                    
                    if matching_results:
                        st.write(f"Found {len(matching_results)} potential matches:")
                        
                        for i, result in enumerate(matching_results):
                            customer = result['customer']
                            match_percentage = round(result['avg_score'])
                            
                            st.markdown(f"### Match #{i+1} - {match_percentage}% overall similarity")
                            st.write(f"**Database ID:** {customer['id']}")
                            st.write(f"**Name:** {customer['name']}")
                            st.write(f"**Phone:** {customer['phone']}")
                            st.write(f"**Address:** {customer['address']}")
                            st.write(f"**Current Status:** {customer['verification_status']}")
                            
                            # Display match details
                            st.markdown("**Match Details:**")
                            for field, detail in result['match_details'].items():
                                icon = "✅" if field in result['matches'] else "❌"
                                st.write(f"{icon} {field.title()}: {detail}")
                                
                            # Option to verify this customer
                            if st.button(f"Verify Customer #{customer['id']}"):
                                update_verification_status(customer['id'], "verified")
                                st.success(f"Customer #{customer['id']} has been verified!")
                    else:
                        st.warning("No matching records found in the database.")
                        
                        # Option to add this customer to the database
                        if st.button("Add as New Customer"):
                            customer_id = add_customer(
                                extracted_info['name'],
                                extracted_info['phone'],
                                extracted_info['address']
                            )
                            st.success(f"New customer added with ID #{customer_id}")
                
                finally:
                    # Cleanup
                    if 'file_path' in locals():
                        try:
                            os.remove(file_path)
                        except:
                            pass
    
    # Option to view all customers in the database
    if st.checkbox("View All Customers in Database"):
        customers = get_all_customers()
        
        if customers:
            st.subheader(f"All Customers ({len(customers)})")
            for customer in customers:
                st.markdown(f"""
                **ID:** {customer['id']}  
                **Name:** {customer['name']}  
                **Phone:** {customer['phone']}  
                **Address:** {customer['address']}  
                **Status:** {customer['verification_status']}
                ---
                """)
        else:
            st.info("No customers in the database yet.")

if __name__ == "__main__":
    main()
