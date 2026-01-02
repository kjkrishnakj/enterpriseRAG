from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import RAGConfig
import torch

MODEL_NAME = RAGConfig.MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate(context, question):
    if not context.strip():
        return "I don't know"

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say exactly: I don't know.

Context:
{context}

Question:
{question}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=RAGConfig.MAX_NEW_TOKENS,
            temperature=0
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer if answer else "I don't know"
