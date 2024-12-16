import torch
import streamlit as st
from lightning_module import extract_triplets
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    
    text = st.text_input('Input text', 'Choulex is a commune in the canton of Geneva, it borders Vandoeuvres and Thônex.')
    num_beams = st.slider('num_beams', 1, 12, 3)
    num_ret_seq = st.slider('num_ret_seq', 1, num_beams, 1)
   
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": num_beams,
        "num_return_sequences": num_ret_seq,
    }
    
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
    
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        **gen_kwargs,
    )
    
    
    decoded_preds: list[str] = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    
    for idx, sentence in enumerate(decoded_preds):
        st.title(f"Predicted triplets for sentence {idx}: ")
        st.write(extract_triplets(sentence))