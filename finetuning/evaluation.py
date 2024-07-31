import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

# Load models and tokenizers
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
original_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

fine_tuned_model = AutoModelForCausalLM.from_pretrained('./quantized_fine_tuned_llama')
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('./quantized_fine_tuned_llama')

# Function to generate QA pairs
def generate_qa_pairs(model, tokenizer, context, max_length=150):
    input_ids = tokenizer.encode(context, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Select a context
context = "Your chosen context here..."

# Generate QA pairs
original_qa_pairs = generate_qa_pairs(original_model, tokenizer, context)
fine_tuned_qa_pairs = generate_qa_pairs(fine_tuned_model, fine_tuned_tokenizer, context)

# Load metrics
bleu_metric = load_metric('bleu')
rouge_metric = load_metric('rouge')

# Function to evaluate metrics
def evaluate_metrics(original_text, generated_text):
    bleu_score = bleu_metric.compute(predictions=[generated_text], references=[[original_text]])
    rouge_score = rouge_metric.compute(predictions=[generated_text], references=[[original_text]])
    return bleu_score, rouge_score

# Evaluate metrics
original_bleu, original_rouge = evaluate_metrics(context, original_qa_pairs)
fine_tuned_bleu, fine_tuned_rouge = evaluate_metrics(context, fine_tuned_qa_pairs)

# Print results
print("Original Model BLEU Score:", original_bleu)
print("Original Model ROUGE Score:", original_rouge)
print("Fine-Tuned Model BLEU Score:", fine_tuned_bleu)
print("Fine-Tuned Model ROUGE Score:", fine_tuned_rouge)

# Save results to JSON
results = {
    "context": context,
    "original_model": {
        "qa_pairs": original_qa_pairs,
        "metrics": {
            "bleu": original_bleu,
            "rouge": original_rouge
        }
    },
    "fine_tuned_model": {
        "qa_pairs": fine_tuned_qa_pairs,
        "metrics": {
            "bleu": fine_tuned_bleu,
            "rouge": fine_tuned_rouge
        }
    }
}

with open('qa_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Evaluation results saved to 'qa_evaluation_results.json'.")
