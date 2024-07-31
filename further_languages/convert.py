import datasets
import torch
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransTokenizer import IndicProcessor
from tqdm import tqdm

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def translate_squad(squad_dataset, src_lang, tgt_lang, model_path, batch_size=32, max_samples=100):
  model, tokenizer = load_model_and_tokenizer(model_path)
  ip = IndicProcessor(inference=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  translated_data = []

  # Access the 'train' dataset
  train_dataset = squad_dataset['train']

  # Iterate over the dataset in batches
  for i in tqdm(range(0, min(max_samples, len(train_dataset)), batch_size)):
    batch = train_dataset[i:i+batch_size]  # Access data using indexing

    contexts = batch["context"]
    questions = batch["question"]
    answers = [["text"][0] for answer in batch["answers"]]  # Extract answer text

    try:
      all_texts = contexts + questions + answers
      print(f"Translating batch {i//batch_size + 1}")
      print(f"Number of texts to translate: {len(all_texts)}")
      print(f"First text to translate: {all_texts[0][:100]}...")

      translated_texts = translate_batch(all_texts, src_lang, tgt_lang, model, tokenizer, ip, device)

      print(f"Number of translated texts: {len(translated_texts)}")
      print(f"First translated text: {translated_texts[0][:100]}...")

      translated_contexts = translated_texts[:len(contexts)]
      translated_questions = translated_texts[len(contexts):len(contexts)+len(questions)]
      translated_answers = translated_texts[len(contexts)+len(questions):]

      for j in range(len(contexts)):
        translated_example = {
          "context": translated_contexts[j],
          "question": translated_questions[j],
          "answers": {
            "text": [translated_answers[j]],
            "answer_start": []  # Note: answer_start positions are lost in translation
          }
        }
        translated_data.append(translated_example)

    except Exception as e:
      print(f"Error in batch {i//batch_size + 1}: {str(e)}")
      print(f"Problematic batch: {batch}")
      continue

  return translated_data

def translate_batch(texts, src_lang, tgt_lang, model, tokenizer, ip, device, max_length=256):
    try:
        batch = ip.preprocess_batch(texts, src_lang, tgt_lang)
        inputs = tokenizer(batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, num_beams=5, num_return_sequences=1, max_length=max_length)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return decoded
    except Exception as e:
        print(f"Error in translate_batch: {str(e)}")
        print(f"First text in batch: {texts[0][:100]}...")  # Print first 100 chars of first text
        raise  # Re-raise the exception to be caught in the calling function

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Main execution
try:
    squad_dataset = datasets.load_dataset("squad")
    src_lang = "eng_Latn"
    tgt_lang = "pan_Guru"
    model_path = "/home/ubuntu/harsh/harsh_project/model/ai4bharat/indictrans2-en-indic-1B"
    output_file = "squad_punjabi_100.json"

    # Access the 'train' dataset
    train_dataset = squad_dataset['train']
    validation_dataset = squad_dataset["validation"]
    print(squad_dataset.column_names)
    print(train_dataset.column_names)
    
    
    print(f"Dataset information:")
    print(f"Type of squad_dataset: {type(squad_dataset)}")
    print(f"Keys in squad_dataset: {squad_dataset.keys()}")
    print(f"Type of train_dataset: {type(train_dataset)}")
    print(f"Column names in train_dataset: {train_dataset.column_names}")
    print(f"Number of examples in train_dataset: {len(train_dataset)}")
    
    # Similarly for validation dataset
    print(f"Type of validation_dataset: {type(validation_dataset)}")
    print(f"Column names in validation_dataset: {validation_dataset.column_names}")
    print(f"Number of examples in validation_dataset: {len(validation_dataset)}")

    translated_data = translate_squad(squad_dataset, src_lang, tgt_lang, model_path, max_samples=100)
    save_to_json(translated_data, output_file)
    print(f"Translation completed. Output saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {str(e)}")