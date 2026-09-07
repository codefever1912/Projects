# ai_module/scorer.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained LLaMA model
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example, replace with specific model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def calculate_risk_score(scan_results):
    score = 0
    
    # Simple scoring based on open ports, subdomains, and expired SSL
    if scan_results.get("open_ports"):
        score += 20 * len(scan_results["open_ports"])
    if scan_results.get("subdomains"):
        score += 10 * len(scan_results["subdomains"])
    if scan_results.get("expired_ssl"):
        score += 50

    # Risk score calculated from model output (can be a more complex logic later)
    input_text = f"Scan results: {scan_results}. What is the risk score?"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract numeric score (assumption: the model outputs a numeric risk score)
    # For now, we keep the base scoring system, but you can replace it with model interpretation
    risk_score_from_model = int(generated_text.split()[-1]) if generated_text[-1].isdigit() else 0
    return min(score + risk_score_from_model, 100)


# ai_module/summarizer.py
def summarize_risk(domain, scan_results):
    subdomains = scan_results.get("subdomains", [])
    ports = scan_results.get("open_ports", [])
    expired_ssl = scan_results.get("expired_ssl", False)
    
    # Basic information generation
    summary = f"Risk Summary for {domain}:\n"
    if subdomains:
        summary += f"- {len(subdomains)} subdomains found.\n"
    if ports:
        summary += f"- {len(ports)} open ports detected.\n"
    if expired_ssl:
        summary += f"- Expired SSL certificate detected.\n"

    # Using LLaMA model for a more descriptive risk summary
    input_text = f"Scan results: {summary}. Please provide a detailed risk summary."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text
