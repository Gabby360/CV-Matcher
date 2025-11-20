# CV Matcher

CV Matcher is a Streamlit-based machine learning app that helps recruiters rank candidate CVs by how well they match a job requirements document. It uses a SentenceTransformer model to compute semantic similarity between the job requirements and each candidate's CV.

## Features

- Upload job requirements (PDF, DOCX, or TXT)
- Upload multiple candidate CVs (PDF, DOCX, or TXT)
- Ranks candidates by similarity to the requirements
- Displays a ranking table with match percentages and positions (1st, 2nd, etc.)
- Visualizes candidate match percentages with a bar chart
- Preview the extracted text of each CV

## Setup

### 1. Clone or Download the Project

Place all files in a folder, e.g. `C:\Users\user\Desktop\CV Matcher`.

### 2. Create and Activate a Virtual Environment

```cmd
cd "C:\Users\user\Desktop\CV Matcher"
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements

```cmd
pip install -r requirements.txt
```

### 4. Run the App

```cmd
streamlit run app.py
```

Open the provided URL (usually http://localhost:8501) in your browser.

## Usage

1. Upload a job requirements file.
2. Upload one or more candidate CVs.
3. View the ranked table and bar chart.
4. Expand each candidate to preview their CV text.

## Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies.

## Customization

- You can change the SentenceTransformer model in the sidebar.
- To improve extraction for scanned PDFs, consider adding OCR support.

## Stopping the App

Press `Ctrl+C` in the terminal to stop the Streamlit server.

---

**For questions or improvements, feel free to open an issue or contact the author.**
