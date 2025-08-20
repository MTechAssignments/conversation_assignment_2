
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time

# -------- Config --------
#MODEL_PATH = "./model/gpt2-finetuned-model"  # local directory where model is saved
MODEL_PATH = "Anup77Jindal/gpt2-finetuned-model"


def download_model():
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

        start_time = time.time()
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_new_tokens).to(model.device)

        outputs = model.generate(
        **inputs,
        max_length=128 + inputs.input_ids.shape[1], # Increase max_length to include prompt
        return_dict_in_generate=True,
        output_scores=True # Keep output_scores to calculate confidence
        )        
        inference_time = time.time() - start_time

        # Decode the generated answer
        generated_sequence = outputs.sequences[0]
        #print(f"generated_sequence: {generated_sequence}\n\n")
        
        # Get the length of the input prompt's token IDs
        prompt_length = inputs.input_ids.shape[1]
        
        # Slice the generated sequence to get only the generated answer part
        answer_ids = generated_sequence[prompt_length:]
        #print(f"answer_ids: {answer_ids}\n\n") 
        
        decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        # Calculate confidence score from the transition scores of the generated tokens
        # We calculate the average probability of the generated tokens
        # The scores are the logits of the next token predicted
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        # Calculate the average log probability across generated tokens
        avg_log_prob = transition_scores.mean().item()
        # Exponentiate the average log probability to get a probability-like score
        confidence = torch.exp(torch.tensor(avg_log_prob)).item()

        data = {
        "Question": query,
        "Method": "Fine-Tune",
        "Answer": decoded_answer,
        "Confidence": f"{confidence:.4f}",
        "Time (s)": f"{inference_time:.2f}"
        }

        return data
        

    # return the callable so all calls share same guardrails/params
    return generate_answer

