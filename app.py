import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("NANONETS_API_KEY")

# Streamlit UI
st.title("📄 Nanonets OCR Demo")
st.write("Upload an image and extract text using the Nanonets OCR API.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Extract Text"):
        url = "https://app.nanonets.com/api/v2/OCR/Model/"

        files = {'file': uploaded_file.getvalue()}
        data = {
            "categories": ["default"],
            "model_type": "ocr"
        }

        with st.spinner("🔍 Extracting text from image..."):
            response = requests.post(
                url,
                auth=requests.auth.HTTPBasicAuth(api_key, ''),
                files=files,
                data={"data": str(data)}
            )

        if response.status_code == 200:
            result = response.json()
            st.success("✅ OCR Text Extracted!")
            st.json(result)
        else:
            st.error(f"❌ Error {response.status_code}")
            st.text(response.text)
