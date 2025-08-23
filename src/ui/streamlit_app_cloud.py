# Check if the app is running on Streamlit Community Cloud
if os.getenv("STREAMLIT_SERVER_RUN_ON_SAVE", "false").lower() == "true":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
import os
import sys
import streamlit as st
import requests
import pandas as pd

# importing necessary functions from dotenv library
from dotenv import load_dotenv  

load_dotenv('config/.env') 

# Get the path of the parent directory (conversation_assignment_2)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from  src.model import inference as infer
from src.api import rag_server as rags

MODE_FINE_TUNE_MODEL = "Use Fine-tuned Model"
MODE_RAG = "Use RAG"

RAG_SERVER_API_URI = f"http://localhost:8000/rag"

@st.cache_resource
def load_model():
    return infer.download_model()

def create_dataframe(data):
    if isinstance(data, list):
        # convert List of dicts into dataframe
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Dict of scalars/lists → try single-row frame
        # If any value is a list of equal length, DataFrame will tabulate columns
        # Otherwise, make it a single-row table
        if any(isinstance(v, list) for v in data.values()):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        df = pd.DataFrame([{"response": str(data)}])

    return df

# -------- UI --------
st.title("Invoke RAG or Fine-tuned Model for Financial Question Answers")

# Inputs
user_input = st.text_area("Query", "Welcome to GE healthcare. Please enter your query.")
max_length = 512

mode = st.radio(
    "Choose an Action",
    [MODE_FINE_TUNE_MODEL, MODE_RAG ],
    horizontal=True
)

# Submit
if st.button("Submit"):
    if not user_input.strip():
        st.error("This field is mandatory. Please enter a valid query.")
    else:
        st.success("You entered: " + user_input)

        if mode == MODE_RAG:
            # ---- Call Flask endpoint ----
            payload = {
                "query": user_input,
                "max_length": max_length
            }
            try:
                with st.spinner("Calling Flask API..."):
                     
                    data = rags.invoke_rag(user_input, max_length)
                    #resp = requests.post(RAG_SERVER_API_URI, json=payload, timeout=60)
                #if not resp.ok:
                if not data:
                    #st.error(f"API error {resp.status_code}: {resp.text}")
                    st.error(f"No response returned from RAG System")
                else:
                    #data = resp.json()
                    #st.subheader("API Response (JSON)")
                    st.json(data)

                    # ---- Show as table when possible ----
                    st.subheader("RAG Response:")
                    try:
                        df = create_dataframe(data) 
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.info("Failed to render the response as table; showing JSON as-is.")
                        st.json(data)
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to reach Flask API: {e}")

        elif mode == MODE_FINE_TUNE_MODEL:
             # ---- Use the fine-tuned model ----
              with st.spinner("Generating with fine-tuned model..."):
                try:
                    generate_answer = load_model()
                    data = generate_answer(user_input, max_new_tokens=max_length)
                    st.subheader("Finetuned Response:")
                    
                    df = create_dataframe(data) 
                    st.dataframe(df, use_container_width=True)
                    #st.info(data)
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")