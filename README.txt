

Virtual Environment Setup

# Create venv
python -m venv venv
# Activate
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on Mac/Linux

# Install dependencies
pip install -r requirements.txt

After activating the venv, run this
pip install streamlit-extras
The command prompt should look like: "(venv) PS C:\Projects\rag-chatbot> pip install streamlit-extras".

Setup the API Keys

#Create a .env file in the root folder.
Sample:
GOOGLE_API_KEY=google_token_here
HF_TOKEN=hf_token_here

#Run the app:
streamlit run app.py


