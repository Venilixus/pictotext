import os
import io
import json
import re
import hashlib
import requests
from dotenv import load_dotenv
import streamlit as st
from PIL import Image, UnidentifiedImageError
import pytesseract
import pandas as pd
import openai
from pdf2image import convert_from_bytes
import time
import email
from email import policy
from email.parser import BytesParser

# Load environment variables (if any)
load_dotenv()

# Explicitly set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Sidebar: OpenAI API Key Input
api_key_input = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if api_key_input:
    openai.api_key = api_key_input
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

###############################
# Helper Functions
###############################

def get_openai_credit():
    """
    Attempts to fetch remaining credit via an unofficial OpenAI billing endpoint.
    Note: This endpoint is unofficial and may fail.
    """
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    url = "https://api.openai.com/dashboard/billing/credit_grants"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            total_granted = data.get("total_granted", 0)
            total_used = data.get("total_used", 0)
            return total_granted - total_used
        else:
            return None
    except Exception as e:
        st.error(f"Error retrieving credit info: {e}")
        return None

def save_correction(file_name, original_text, correction, corrections_path="corrections.csv"):
    """
    Saves feedback corrections to a CSV file.
    """
    new_entry = pd.DataFrame({
        "file_name": [file_name],
        "original_text": [original_text],
        "correction": [correction]
    })
    if os.path.exists(corrections_path):
        existing_df = pd.read_csv(corrections_path)
        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
    else:
        updated_df = new_entry
    updated_df.to_csv(corrections_path, index=False)

def compute_file_hash(file_obj):
    """
    Computes an MD5 hash of the file's contents to uniquely identify the file.
    """
    file_obj.seek(0)
    file_bytes = file_obj.read()
    file_obj.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

def is_valid_image(file_obj):
    """
    Checks if the uploaded file is a valid image.
    """
    try:
        file_obj.seek(0)
        img = Image.open(file_obj)
        img.verify()  # Validate image integrity
        file_obj.seek(0)
        return True
    except Exception:
        file_obj.seek(0)
        return False

def extract_text(image_file):
    """
    Extracts text from an image using pytesseract.
    """
    image_file.seek(0)
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def parse_eml(file_obj):
    """
    Parses an .eml file and extracts plain text content.
    """
    file_obj.seek(0)
    msg = BytesParser(policy=policy.default).parse(file_obj)
    text_content = ""
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == "text/plain":
                text_content += part.get_content()
    else:
        if msg.get_content_type() == "text/plain":
            text_content = msg.get_content()
    return text_content

def parse_ocr_to_structured_data(ocr_text):
    """
    Uses ChatGPT (gpt-3.5-turbo) to parse OCR text into structured data.
    Returns a DataFrame with columns: date, price, metal_type, provider, description.
    Date should be in dd-mm-yyyy format.
    """
    prompt = f"""
    You are a helpful assistant that parses text from an OCR scan.
    The text might contain one or more line items of a purchase or inventory record.
    
    Please extract each line item and return a valid JSON array of objects.
    Each object must have these keys:
      - "date": a string (or blank if missing) in the format dd-mm-yyyy
      - "price": a numeric or string value (or blank if missing)
      - "metal_type": a string (or blank if missing)
      - "provider": a string (or blank if missing)
      - "description": a string summarizing the item or line
    
    If any fields are not present in the text, leave them as an empty string.
    Example output:
    [
      {{
        "date": "01-10-2023",
        "price": "19.99",
        "metal_type": "Steel",
        "provider": "ACME Metals",
        "description": "Metal rod 10x"
      }}
    ]
    
    Here is the text to parse:
    \"\"\"{ocr_text}\"\"\"
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
        else:
            st.warning("No valid line items found. Using empty DataFrame.")
            df = pd.DataFrame(columns=["date", "price", "metal_type", "provider", "description"])
        for col in ["date", "price", "metal_type", "provider", "description"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON from ChatGPT: {e}")
        return pd.DataFrame(columns=["date", "price", "metal_type", "provider", "description"])
    except Exception as e:
        st.error(f"Error calling ChatGPT: {e}")
        return pd.DataFrame(columns=["date", "price", "metal_type", "provider", "description"])

def extract_numeric_price(price_str):
    """
    Extracts a numeric value from a price string using regex.
    Returns a float.
    """
    match = re.search(r'[\d,.]+', price_str)
    if match:
        num_str = match.group(0)
        if ',' in num_str and '.' in num_str:
            num_str = num_str.replace(',', '')
        elif ',' in num_str and '.' not in num_str:
            num_str = num_str.replace(',', '.')
        try:
            return float(num_str)
        except:
            return 0.0
    else:
        return 0.0

def clean_and_format_data(df):
    """
    Cleans and formats the DataFrame:
      - Converts dates to dd-mm-yyyy.
      - Processes the price field into three columns:
            Interpreted price, Price per kilo, Price per MT.
          If the price string contains "MT" (case-insensitive), it is assumed to be per metric ton;
          otherwise, it is assumed to be per kilogram.
      - Drops the original 'price' column.
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
        df['date'] = df['date'].dt.strftime('%d-%m-%Y')
        df['date'] = df['date'].fillna('')
    if 'price' in df.columns:
        df['price'] = df['price'].astype(str)
        df['price'] = df['price'].str.replace('€', '', regex=False)
        df['price'] = df['price'].str.strip()
        
        def process_price_cell(price_str):
            numeric_value = extract_numeric_price(price_str)
            if "mt" in price_str.lower():
                return numeric_value, numeric_value / 1000.0, numeric_value
            else:
                return numeric_value, numeric_value, numeric_value * 1000.0
        
        processed = df['price'].apply(process_price_cell)
        df['Interpreted price'] = processed.apply(lambda x: x[0])
        df['Price per kilo'] = processed.apply(lambda x: x[1])
        df['Price per MT'] = processed.apply(lambda x: x[2])
        df = df.drop(columns=['price'])
    return df

def append_to_output_df(new_data):
    """
    Appends new data to the output DataFrame stored in session state.
    Reorders columns so that the 'source' column (if present) is the first column.
    """
    new_data = clean_and_format_data(new_data)
    if 'output_df' not in st.session_state or st.session_state.output_df.empty:
        st.session_state.output_df = new_data
    else:
        st.session_state.output_df = pd.concat([st.session_state.output_df, new_data], ignore_index=True)
    if 'source' in st.session_state.output_df.columns:
        cols = st.session_state.output_df.columns.tolist()
        cols.remove('source')
        new_order = ['source'] + cols
        st.session_state.output_df = st.session_state.output_df[new_order]
    return st.session_state.output_df

def get_excel_file_bytes():
    """
    Converts the output DataFrame in session state to an Excel file in memory.
    Returns a BytesIO object.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.output_df.to_excel(writer, index=False)
    output.seek(0)
    return output

def show_statistics(excel_df):
    """
    Displays summary statistics based on the given DataFrame.
    """
    st.subheader("Overall Summary Statistics")
    st.write("Total records:", len(excel_df))
    if "Interpreted price" in excel_df.columns:
        st.write("Average Interpreted Price:", excel_df["Interpreted price"].mean())
    if "Price per kilo" in excel_df.columns:
        st.write("Average Price per kilo:", excel_df["Price per kilo"].mean())
    if "Price per MT" in excel_df.columns:
        st.write("Average Price per MT:", excel_df["Price per MT"].mean())
    st.subheader("Records by Provider")
    if "provider" in excel_df.columns:
        st.dataframe(excel_df.groupby("provider").size().reset_index(name="Count"))

###############################
# Main App: Page Selection
###############################

page = st.sidebar.selectbox("Select Page", ["Process Files", "Statistics"])

if page == "Process Files":
    st.title("Pictotext OCR & ChatGPT App (Structured Data)")
    
    remaining_credit = get_openai_credit()
    if remaining_credit is not None:
        st.sidebar.info(f"Remaining OpenAI Credit: €{remaining_credit:.2f}")
    else:
        st.sidebar.warning("Unable to retrieve OpenAI credit information.")
    
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=["png", "jpg", "jpeg", "pdf", "csv", "eml"],
        accept_multiple_files=True
    )
    
    if 'output_df' not in st.session_state:
        st.session_state.output_df = pd.DataFrame()
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_name_lower = file_name.lower()
            if file_name_lower.endswith((".png", ".jpg", ".jpeg")):
                if is_valid_image(uploaded_file):
                    st.image(uploaded_file, caption=f"Uploaded: {file_name}", use_container_width=True)
                else:
                    st.error(f"Error displaying {file_name}: Cannot identify image file.")
            else:
                st.info(f"Uploaded file: {file_name}")
    
        parse_line_items = st.checkbox("Parse OCR text into structured data", value=True)
    
        if st.button("Process Files"):
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            file_counter = 0
    
            processed_file_ids = set()
            if not st.session_state.output_df.empty and 'file_id' in st.session_state.output_df.columns:
                processed_file_ids = set(st.session_state.output_df['file_id'].dropna().unique())
    
            for uploaded_file in uploaded_files:
                file_counter += 1
                status_text.text(f"Processing file {file_counter} of {total_files}: {uploaded_file.name}")
                file_name = uploaded_file.name
                file_name_lower = file_name.lower()
                file_id = compute_file_hash(uploaded_file)
    
                if file_id in processed_file_ids:
                    st.warning(f"File {file_name} already processed. Skipping duplicate.")
                    progress_bar.progress(file_counter / total_files)
                    continue
    
                extracted_text = ""
                if file_name_lower.endswith('.pdf'):
                    try:
                        pdf_bytes = uploaded_file.read()
                        pages = convert_from_bytes(pdf_bytes)
                        full_text = ""
                        for page in pages:
                            buf = io.BytesIO()
                            page.save(buf, format="JPEG")
                            buf.seek(0)
                            full_text += "\n" + extract_text(buf)
                        extracted_text = full_text
                    except Exception as e:
                        st.error(f"Error processing PDF {file_name}: {e}")
                        continue
                elif file_name_lower.endswith('.csv'):
                    try:
                        df_csv = pd.read_csv(uploaded_file)
                        for col in ["date", "price", "metal_type", "provider", "description"]:
                            if col not in df_csv.columns:
                                df_csv[col] = ""
                        df_csv["source"] = file_name
                        df_csv["file_id"] = file_id
                        st.session_state.output_df = append_to_output_df(df_csv)
                        st.success(f"CSV file {file_name} processed and appended to output.")
                        progress_bar.progress(file_counter / total_files)
                        continue
                    except Exception as e:
                        st.error(f"Error reading CSV {file_name}: {e}")
                        continue
                elif file_name_lower.endswith('.eml'):
                    try:
                        extracted_text = parse_eml(uploaded_file)
                    except Exception as e:
                        st.error(f"Error processing EML {file_name}: {e}")
                        continue
                else:
                    if not is_valid_image(uploaded_file):
                        st.error(f"Skipping {file_name}: Not a valid image.")
                        progress_bar.progress(file_counter / total_files)
                        continue
                    extracted_text = extract_text(uploaded_file)
    
                if parse_line_items:
                    with st.spinner(f"Parsing OCR text for {file_name}..."):
                        df_line_items = parse_ocr_to_structured_data(extracted_text)
                    df_line_items["source"] = file_name
                    df_line_items["file_id"] = file_id
                    st.subheader(f"Parsed Line Items from {file_name}:")
                    st.dataframe(df_line_items)
                    st.session_state.output_df = append_to_output_df(df_line_items)
                else:
                    df_raw = pd.DataFrame({
                        "date": [""],
                        "Interpreted price": [""],
                        "Price per kilo": [""],
                        "Price per MT": [""],
                        "description": [extracted_text],
                        "source": [file_name],
                        "file_id": [file_id]
                    })
                    st.session_state.output_df = append_to_output_df(df_raw)
    
                st.subheader(f"Feedback for {file_name}")
                correction = st.text_area(f"Enter correction for {file_name} (if any):", key=file_name)
                if st.button(f"Submit Correction for {file_name}", key=file_name + "-button"):
                    save_correction(file_name, extracted_text, correction)
                    st.success(f"Correction for {file_name} submitted.")
    
                progress_bar.progress(file_counter / total_files)
                time.sleep(0.5)
                
            status_text.text("Processing complete!")
            st.success("Data processed!")
            excel_file = get_excel_file_bytes()
            st.download_button("Download Excel File", excel_file, file_name="output.xlsx")
    
elif page == "Statistics":
    st.title("Statistics")
    if 'output_df' in st.session_state and not st.session_state.output_df.empty:
        show_statistics(st.session_state.output_df)
    else:
        st.warning("No data available. Process some files first.")
