import streamlit as st
import requests
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# -------- Config --------
#MODEL_PATH = "./model/gpt2-finetuned-model"  # local directory where model is saved
MODEL_PATH = "Anup77Jindal/gpt2-finetuned-model"

DEFAULT_FLASK_URL = API_URL = "http://localhost:8000/rag"  # change to RAG endpoint

MODE_FINE_TUNE_MODEL = "Use Fine-tuned Model"
MODE_RAG = "Use RAG"

@st.cache_resource
def load_model():
    """
    Loads the fine-tuned GPT-2 model + tokenizer and returns a
    callable `generate_answer(query, max_new_tokens)` that applies:
      - guardrails on input
      - a financial QA prompt template
      - safe decoding parameters
    """
    # --- Load model/tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # GPT-2 has no PAD token by default; tie pad to EOS for batching/pipe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # helpful for causal LMs on long prompts

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # --- lightweight guardrails ---
    FINANCE_KEYWORDS = {
        "revenue","profit","loss","operating","opex","capex","income","cash",
        "margin","ebitda","ebit","guidance","forecast","unit","product",
        "segment","q1","q2","q3","q4","fy","year","quarter","customers",
        "ge healthcare","financial","price","sales","growth","cost","expense",
        "balance sheet","inventory","backlog"
    }
    DENY_PATTERNS = [
        r"\b(ssn|social\s*security|password|credit\s*card|cvv)\b",
        r"\b(api[_-]?key|token)\b",
        r"(suicide|self-harm)"
    ]

    def is_out_of_scope(q: str) -> bool:
        # very light “domain” heuristic: if not a single finance keyword hits, consider OOS
        qlow = q.lower()
        return not any(k in qlow for k in FINANCE_KEYWORDS)

    def is_unsafe(q: str) -> bool:
        import re
        qlow = q.lower()
        return any(re.search(p, qlow) for p in DENY_PATTERNS)

    # --- prompt template to steer the model ---
    SYSTEM = (
        "You are a precise financial question answering assistant for "
        "GE Healthcare. Answer ONLY if the question is about finance or product metrics "
        "(e.g., revenue, units, margins). If the answer is unknown or outside scope, reply "
        "exactly: 'I don't know based on my data.' Respond concisely with numbers and units."
    )
    def build_prompt(question: str) -> str:
        #return f"{SYSTEM}\n\nQuestion: {question}\nAnswer:"   
        return f"question: {question} answer:"

    # --- callable that encapsulates guardrails + generation ---
    def generate_answer(query: str, max_new_tokens: int = 512) -> str:
        q = (query or "").strip()
        if not q:
            return "This field is mandatory. Please enter a valid query."
        if len(q) > 1500:
            return "Your question is too long; please shorten it."
        if is_unsafe(q):
            return "I can’t help with that request."
        if is_out_of_scope(q):
            # keep behavior consistent with RAG’s safety posture
            return "I don't know based on my data."

        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_new_tokens).to(model.device)

        out = model.generate(
        **inputs,
        max_length=128 + inputs.input_ids.shape[1], # Increase max_length to include prompt
        return_dict_in_generate=True,
        output_scores=True # Keep output_scores to calculate confidence
        )        
        
        # Decode the generated answer
        generated_sequence = out.sequences[0]
        #print(f"generated_sequence: {generated_sequence}\n\n")
        
        # Get the length of the input prompt's token IDs
        prompt_length = inputs.input_ids.shape[1]
        
        # Slice the generated sequence to get only the generated answer part
        answer_ids = generated_sequence[prompt_length:]
        #print(f"answer_ids: {answer_ids}\n\n") 
        
        decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return decoded_answer
        

    # return the callable so all calls share same guardrails/params
    return generate_answer




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