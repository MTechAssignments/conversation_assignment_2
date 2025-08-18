import sys
import os

import streamlit as st
import requests
import pandas as pd


# Get the path of the parent directory (conversation_assignment_2)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from  src.model import infer


DEFAULT_FLASK_URL = API_URL = "http://localhost:8000/rag"  # change to RAG endpoint

MODE_FINE_TUNE_MODEL = "Use Fine-tuned Model"
MODE_RAG = "Use RAG"

@st.cache_resource
def load_model():
    return infer.download_model()

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

# Sidebar config for API mode
flask_url = DEFAULT_FLASK_URL 

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
                    resp = requests.post(flask_url, json=payload, timeout=60)
                if not resp.ok:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                else:
                    data = resp.json()
                    st.subheader("API Response (JSON)")
                    st.json(data)

                    # ---- Show as table when possible ----
                    st.subheader("API Response:")
                    # Try to coerce JSON to a DataFrame sensibly
                    try:
                        if isinstance(data, list):
                            # List of dicts → table directly
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
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.info("Could not tabulate the response; showing JSON above.")
                        st.caption(f"(Table error: {e})")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to reach Flask API: {e}")

        elif mode == MODE_FINE_TUNE_MODEL:
             # ---- Use the fine-tuned model ----
              with st.spinner("Generating with fine-tuned model..."):
                generate_answer = load_model()
                answer = generate_answer(user_input, max_new_tokens=max_length)
                st.subheader("Generated Answer")
                st.info(answer)