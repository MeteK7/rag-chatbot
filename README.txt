Virtual Environment Setup

# Create venv
python -m venv venv
# Activate
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on Mac/Linux

# Install dependencies
pip install -r requirements.txt

streamlit run app.py


After activating the venv, run this


pip install streamlit-extras
The command prompt should look like: "(venv) PS C:\Projects\rag-chatbot> pip install streamlit-extras".