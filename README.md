Below is a sample README.md file that you can include in your GitHub repository:

Pictotext OCR & ChatGPT App (Structured Data)

Pictotext is a robust Streamlit application that extracts text from various file types using OCR, structures and cleans the extracted data with the help of OpenAI’s ChatGPT, and compiles everything into a single, standardized Excel file. The app is designed to support multiple file formats, deduplicate entries, and display useful statistics—all via a simple, user-friendly web interface.

Features
	•	Multi-format Support:
	•	Images: PNG, JPG, JPEG (OCR via pytesseract)
	•	PDFs: Converts pages to images using pdf2image (requires Poppler)
	•	CSV Files: Directly reads and appends data
	•	EML Files: Extracts plain text from email files
	•	Data Parsing & Standardization:
	•	Uses ChatGPT (gpt-3.5-turbo) to parse OCR text into structured data with fields such as date, interpreted price, price per kilo, price per MT, metal type, provider, and description.
	•	Standardizes dates in dd-mm-yyyy format and normalizes pricing based on whether the price is provided per kilo or per metric ton.
	•	Deduplication:
	•	Computes a unique file hash (MD5) for each uploaded file and skips duplicates if the file has already been processed.
	•	User Feedback:
	•	Allows users to submit corrections for any misparsed data, saving feedback to a separate CSV file.
	•	Statistics Page:
	•	Displays summary statistics and provider counts from the compiled Excel output.
	•	API Key Input:
	•	Provides a sidebar option to enter an OpenAI API key, so any user can run the app without modifying the source code.
	•	Progress Indicators:
	•	A progress bar and status messages keep users informed during processing.

Prerequisites
	•	Python 3.7+
	•	Pip

Required Python Libraries
	•	streamlit
	•	pytesseract
	•	pdf2image
	•	pandas
	•	openai
	•	python-dotenv
	•	requests
	•	Pillow

You can install the required libraries via pip:

pip install streamlit pytesseract pdf2image pandas openai python-dotenv requests Pillow

External Dependencies
	•	Poppler:
Required for converting PDF pages to images.
	•	macOS:

brew install poppler


	•	Linux (Debian/Ubuntu):

sudo apt-get install poppler-utils


	•	Windows:
Download the Poppler binaries and add the bin folder to your system’s PATH.

Installation
	1.	Clone the Repository:

git clone https://github.com/yourusername/pictotext.git
cd pictotext


	2.	(Optional) Create and Activate a Virtual Environment:

python -m venv env
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate


	3.	Install Dependencies:

pip install -r requirements.txt

(You can generate a requirements.txt file by running pip freeze > requirements.txt after installing the required libraries.)

	4.	Configure Environment Variables:
Create a .env file in the root directory with the following content (optional, as you can also enter the API key via the UI):

OPENAI_API_KEY=your_openai_api_key_here



Usage
	1.	Run the App:

streamlit run app.py


	2.	Enter OpenAI API Key:
	•	In the sidebar, use the text input to enter your OpenAI API key (if not already set in the .env file).
	3.	Select a Page:
	•	Process Files:
Upload your files (images, PDFs, CSVs, or EMLs) and choose whether to parse OCR text into structured data. The app will process the files, avoid duplicates, and append the data to an output Excel file.
	•	Statistics:
View summary statistics and data grouped by provider from the compiled Excel output.
	4.	Download Output:
	•	Once processing is complete, use the provided download button to save the updated Excel file.

File Format Details
	•	Images (PNG, JPG, JPEG):
OCR is applied using pytesseract.
	•	PDF Files:
Converted to images using pdf2image (requires Poppler). Each page is processed via OCR.
	•	CSV Files:
Directly read and merged with existing data. Missing columns (date, price, metal_type, provider, description) are added if necessary.
	•	EML Files:
Parsed using Python’s email library to extract plain text content.

Pricing Logic

The app attempts to intelligently interpret the pricing data:
	•	Interpreted Price:
The raw numeric value extracted from the price string.
	•	Price per Kilo:
	•	If the price string mentions “MT”, it’s assumed to be per metric ton and is divided by 1000.
	•	Otherwise, it’s assumed to be per kilogram.
	•	Price per MT:
	•	If the price string mentions “MT”, it remains as is.
	•	Otherwise, it’s multiplied by 1000.

Deduplication
	•	Each file is hashed using MD5.
	•	If a file’s hash (file_id) is already present in the output Excel, the file is skipped to prevent duplicate entries.

Contributing

Contributions are welcome! If you find issues or have suggestions for improvements, please open an issue or submit a pull request.

License

This project is licensed under the MIT License.

