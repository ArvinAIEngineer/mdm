import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import json
import requests
from typing import Dict, Optional
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

# Configuration
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NANONETS_MODEL_ID = os.getenv("NANONETS_MODEL_ID", "")  # You'll need to set this in your .env file

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return the path"""
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    print(f"DEBUG: Saved file to {file_path}")
    return file_path

def perform_ocr_with_nanonets(file_path: str) -> str:
    """Perform OCR using Nanonets API directly"""
    print(f"\nDEBUG: Starting OCR for document at {file_path}")
    
    try:
        # API endpoint for prediction
        url = f"https://app.nanonets.com/api/v2/OCR/Model/{NANONETS_MODEL_ID}/LabelFile/"
        
        # Open the file and send it to Nanonets
        with open(file_path, 'rb') as f:
            data = {'file': f}
            response = requests.post(
                url,
                auth=requests.auth.HTTPBasicAuth(NANONETS_API_KEY, ''),
                files=data
            )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"DEBUG: OCR API request failed with status code {response.status_code}")
            print(f"DEBUG: Response content: {response.text}")
            return ""
        
        # Parse the response
        result = response.json()
        print(f"DEBUG: OCR API response received. Status: {result.get('message')}")
        
        # Extract the text
        extracted_text = ""
        
        # Navigate through the JSON response to get the text
        # The structure may vary based on your model configuration
        try:
            # For general OCR models, text is usually in result -> prediction -> ocr_text
            if 'result' in result and result['result']:
                prediction = result['result'][0]['prediction']
                
                # Check if this is an OCR text model or a structured model
                if 'ocr_text' in prediction:
                    # Simple OCR text model
                    extracted_text = prediction['ocr_text']
                elif 'pages' in prediction:
                    # Structured document model
                    pages = prediction['pages']
                    for page in pages:
                        for line in page.get('lines', []):
                            extracted_text += line.get('text', '') + "\n"
                else:
                    # Try to extract from any potential text fields
                    for field in prediction:
                        if isinstance(field, dict) and 'text' in field:
                            extracted_text += field['text'] + "\n"
            
            print(f"DEBUG: Extracted text length: {len(extracted_text)}")
            if len(extracted_text) < 100:
                print(f"DEBUG: Warning: Very little text extracted. Full text: {extracted_text}")
            else:
                print(f"DEBUG: First 200 chars of extracted text: {extracted_text[:200]}")
                
            # If no text was extracted, try to get the raw OCR result
            if not extracted_text and 'raw' in result:
                print("DEBUG: Using raw OCR result")
                extracted_text = json.dumps(result['raw'])
                
            return extracted_text
                
        except Exception as extract_error:
            print(f"DEBUG: Error extracting text from OCR response: {str(extract_error)}")
            print(f"DEBUG: Response structure: {json.dumps(result, indent=2)[:500]}...")
            return ""
            
    except Exception as e:
        print(f"DEBUG: Error in OCR process: {str(e)}")
        return ""

def get_extraction_prompt(doc_type: str, text: str) -> str:
    """Generate appropriate prompt based on document type"""
    base_prompt = """
    Extract the following information from the text below and return it as a JSON object:
    - name: The full name of the person
    - phone: The phone number (if present)
    - dob: The date of birth (if present)
    - address: The complete address

    Text: {text}

    Important: Look carefully through the entire text for these details. They might appear anywhere in the document.
    The name might be preceded by terms like "Name:", "Customer Name:", etc.
    The address might be preceded by "Address:", "Residence:", "Location:", etc.
    Phone numbers might be in various formats including +91 prefix or 10 digits.
    Date of birth might be in formats like DD/MM/YYYY, MM-DD-YYYY, or written out.

    Return only the JSON object in this format:
    {{
        "name": "extracted name or null",
        "phone": "extracted phone or null",
        "dob": "extracted dob or null",
        "address": "extracted address or null"
    }}
    """
    
    if doc_type == "id":
        base_prompt += "\nNote: This is an ID document. Look for officially stated name, dob, and address."
    elif doc_type == "bank":
        base_prompt += "\nNote: This is a bank statement. Look for account holder details, dob, and registered address."
    
    print(f"DEBUG: Generated prompt for {doc_type} document")
    return base_prompt.format(text=text)

def extract_entities_using_groq(text: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Extract entities from text using Groq's language model"""
    prompt = get_extraction_prompt(doc_type, text)

    try:
        print(f"\nDEBUG: Sending text to Groq for {doc_type} document")
        print(f"DEBUG: First 200 chars of text: {text[:200]}")
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=500,
        )
        
        # Log the raw text and extracted information for debugging
        print(f"\nDEBUG: Processing {doc_type.upper()} document")
        print(f"DEBUG: Raw OCR text length: {len(text)} characters")
        print(f"DEBUG: Raw OCR text sample: {text[:500]}...")
        print(f"DEBUG: Groq response: {response.choices[0].message.content}")
        
        # Parse the response into a dictionary
        try:
            extracted_info = json.loads(response.choices[0].message.content)
            print(f"DEBUG: Parsed JSON from Groq: {extracted_info}")
            return {
                'name': extracted_info.get('name'),
                'phone': extracted_info.get('phone'),
                'dob': extracted_info.get('dob'),
                'address': extracted_info.get('address')
            }
        except json.JSONDecodeError as je:
            print(f"DEBUG: JSON parse error: {str(je)}")
            print(f"DEBUG: Raw response that failed to parse: {response.choices[0].message.content}")
            return {'name': None, 'phone': None, 'dob': None, 'address': None}
            
    except Exception as e:
        print(f"DEBUG: Error processing {doc_type} document: {str(e)}")
        return {'name': None, 'phone': None, 'dob': None, 'address': None}

def process_document(file_path: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Process document using OCR and Groq for entity extraction"""
    try:
        # Get OCR text using Nanonets API directly
        text = perform_ocr_with_nanonets(file_path)
        
        if not text:
            print(f"DEBUG: No text extracted from {doc_type} document")
            return {'name': None, 'phone': None, 'dob': None, 'address': None}
        
        # Extract entities using Groq
        return extract_entities_using_groq(text, doc_type)
    except Exception as e:
        print(f"DEBUG: Error in document processing: {str(e)}")
        return {'name': None, 'phone': None, 'dob': None, 'address': None}

def compare_with_fuzzy_match(str1: str, str2: str, threshold: int = 80) -> bool:
    """Compare two strings using fuzzy matching"""
    if not str1 or not str2:
        print(f"DEBUG: Fuzzy match failed - one or both strings empty")
        return False
    
    # Clean the strings
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    
    # Get the ratio
    ratio = fuzz.ratio(str1, str2)
    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)
    
    # Use the higher of the two scores
    similarity = max(ratio, token_sort_ratio)
    print(f"DEBUG: Fuzzy match - '{str1}' vs '{str2}' - ratio: {ratio}, token sort ratio: {token_sort_ratio}, using: {similarity}")
    
    return similarity >= threshold

def compare_phone_numbers(phone1: str, phone2: str) -> bool:
    """Compare phone numbers by removing all non-digits"""
    if not phone1 or not phone2:
        print(f"DEBUG: Phone comparison failed - one or both numbers empty")
        return False
    
    # Remove all non-digits
    phone1_clean = ''.join(filter(str.isdigit, phone1))
    phone2_clean = ''.join(filter(str.isdigit, phone2))
    
    # If both are empty after cleaning, return False
    if not phone1_clean or not phone2_clean:
        print(f"DEBUG: Phone comparison failed - one or both numbers have no digits")
        return False
    
    # Compare the last 10 digits if longer
    phone1_final = phone1_clean[-10:] if len(phone1_clean) >= 10 else phone1_clean
    phone2_final = phone2_clean[-10:] if len(phone2_clean) >= 10 else phone2_clean
    
    result = phone1_final == phone2_final
    print(f"DEBUG: Phone comparison - '{phone1}' ({phone1_final}) vs '{phone2}' ({phone2_final}) - Match: {result}")
    
    return result

def compare_dob(dob1: str, dob2: str) -> bool:
    """Compare dates of birth"""
    if not dob1 or not dob2:
        print(f"DEBUG: DOB comparison failed - one or both dates empty")
        return False
    
    # Remove any extra spaces and normalize separators
    dob1_clean = dob1.strip().replace('/', '-').replace('.', '-')
    dob2_clean = dob2.strip().replace('/', '-').replace('.', '-')
    
    result = dob1_clean == dob2_clean
    print(f"DEBUG: DOB comparison - '{dob1}' ({dob1_clean}) vs '{dob2}' ({dob2_clean}) - Match: {result}")
    
    return result

def compare_extracted_info(id_info: Dict, bank_info: Dict) -> tuple:
    """Compare extracted information from both documents"""
    matches = []
    mismatches = []
    match_details = {}
    
    print("\nDEBUG: Comparing extracted information")
    print(f"DEBUG: ID info: {id_info}")
    print(f"DEBUG: Bank info: {bank_info}")
    
    # Compare name
    if id_info['name'] and bank_info['name']:
        name_match = compare_with_fuzzy_match(id_info['name'], bank_info['name'])
        if name_match:
            matches.append('name')
            match_details['name'] = "Matched with high confidence"
        else:
            mismatches.append('name')
            match_details['name'] = "Names differ significantly"
    else:
        print(f"DEBUG: Skipping name comparison - ID name: {id_info['name']}, Bank name: {bank_info['name']}")
    
    # Compare phone
    if id_info['phone'] and bank_info['phone']:
        phone_match = compare_phone_numbers(id_info['phone'], bank_info['phone'])
        if phone_match:
            matches.append('phone')
            match_details['phone'] = "Phone numbers match"
        else:
            mismatches.append('phone')
            match_details['phone'] = "Phone numbers differ"
    else:
        print(f"DEBUG: Skipping phone comparison - ID phone: {id_info['phone']}, Bank phone: {bank_info['phone']}")
    
    # Compare dob
    if id_info['dob'] and bank_info['dob']:
        dob_match = compare_dob(id_info['dob'], bank_info['dob'])
        if dob_match:
            matches.append('dob')
            match_details['dob'] = "Dates of birth match"
        else:
            mismatches.append('dob')
            match_details['dob'] = "Dates of birth differ"
    else:
        print(f"DEBUG: Skipping DOB comparison - ID DOB: {id_info['dob']}, Bank DOB: {bank_info['dob']}")
    
    # Compare address
    if id_info['address'] and bank_info['address']:
        address_match = compare_with_fuzzy_match(id_info['address'], bank_info['address'])
        if address_match:
            matches.append('address')
            match_details['address'] = "Addresses match with high similarity"
        else:
            mismatches.append('address')
            match_details['address'] = "Addresses differ significantly"
    else:
        print(f"DEBUG: Skipping address comparison - ID address: {id_info['address']}, Bank address: {bank_info['address']}")
    
    # Consider verified if at least 2 fields match
    is_verified = len(matches) >= 2
    print(f"DEBUG: Verification result: {is_verified} with {len(matches)} matches and {len(mismatches)} mismatches")
    
    return is_verified, matches, mismatches, match_details

def main():
    st.title("Customer Document Verification")
    
    # Display warning if model ID is not set
    if not NANONETS_MODEL_ID:
        st.warning("⚠️ Please set the NANONETS_MODEL_ID in your .env file")
    
    st.write("Please upload your documents for verification:")
    
    id_file = st.file_uploader("Upload ID Document (PDF)", type="pdf", key="id_file")
    bank_file = st.file_uploader("Upload Bank Statement (PDF)", type="pdf", key="bank_file")
    
    # Debug toggle
    show_debug = st.checkbox("Show Debug Information", value=False)
    
    if id_file and bank_file:
        if st.button("Verify Documents"):
            with st.spinner("Processing documents..."):
                try:
                    print("\n=== STARTING NEW VERIFICATION PROCESS ===")
                    # Save and process files
                    id_path = save_uploaded_file(id_file)
                    bank_path = save_uploaded_file(bank_file)
                    
                    # Extract information
                    id_info = process_document(id_path, "id")
                    bank_info = process_document(bank_path, "bank")
                    
                    # Display extracted information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("ID Document Info")
                        st.write(f"Name: {id_info['name'] or 'Not found'}")
                        st.write(f"Phone: {id_info['phone'] or 'Not found'}")
                        st.write(f"DOB: {id_info['dob'] or 'Not found'}")
                        st.write(f"Address: {id_info['address'] or 'Not found'}")
                    
                    with col2:
                        st.subheader("Bank Statement Info")
                        st.write(f"Name: {bank_info['name'] or 'Not found'}")
                        st.write(f"Phone: {bank_info['phone'] or 'Not found'}")
                        st.write(f"DOB: {bank_info['dob'] or 'Not found'}")
                        st.write(f"Address: {bank_info['address'] or 'Not found'}")
                    
                    # Compare and verify
                    is_verified, matches, mismatches, match_details = compare_extracted_info(id_info, bank_info)
                    
                    # Display verification results
                    st.subheader("Verification Results")
                    if is_verified:
                        st.success("Documents Verified Successfully!")
                    else:
                        st.error("Document Verification Failed")
                    
                    # Show detailed matching results
                    st.subheader("Matching Details")
                    
                    for field in ['name', 'phone', 'dob', 'address']:
                        if field in match_details:
                            icon = "✅" if field in matches else "❌"
                            st.write(f"{icon} {field.title()}: {match_details[field]}")
                        else:
                            st.write(f"❓ {field.title()}: Insufficient data for comparison")
                    
                    # Show debug information if enabled
                    if show_debug:
                        st.subheader("Debug Information")
                        st.write("ID Document Raw Info:")
                        st.json(id_info)
                        st.write("Bank Statement Raw Info:")
                        st.json(bank_info)
                
                finally:
                    # Cleanup
                    if 'id_path' in locals():
                        try:
                            os.remove(id_path)
                            print(f"DEBUG: Removed temporary file {id_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to remove {id_path}: {str(e)}")
                    if 'bank_path' in locals():
                        try:
                            os.remove(bank_path)
                            print(f"DEBUG: Removed temporary file {bank_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to remove {bank_path}: {str(e)}")

if __name__ == "__main__":
    main()
