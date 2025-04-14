import streamlit as st
import requests

# Streamlit UI
st.title("Nanonets OCR Demo")
st.write("Upload an image to extract text using Nanonets OCR API.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Extract Text"):
        # API request setup
        url = "https://app.nanonets.com/api/v2/OCR/Model/"
        files = {'file': uploaded_file.getvalue()}
        data = {
            "categories": ["category1", "category2"],
            "model_type": "ocr"
        }
        headers = {
            'accept': "application/json"
        }

        with st.spinner("Sending image to Nanonets OCR..."):
            response = requests.post(
                url,
                auth=requests.auth.HTTPBasicAuth('REPLACE_API_KEY', ''),
                files=files,
                data={"data": str(data)}
            )

        if response.status_code == 200:
            result = response.json()
            st.success("OCR Result:")
            st.json(result)
        else:
            st.error(f"Error: {response.status_code}")
            st.text(response.text)
