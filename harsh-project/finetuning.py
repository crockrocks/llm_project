import json
from transformers import AutoTokenizer, AutoModelForCausalLM,Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes as bnb

try:
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Option 1: Use eos_token as pad_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Option 2: Add new pad_token
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', load_in_8bit=True, device_map='auto')
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)


# Step 2: Prepare Model for k-bit Training (PEFT)
model = prepare_model_for_kbit_training(model)

# Step 3: Set up LoRA configuration
lora_config = LoraConfig(
    r=8,  # The dimension of the LoRA update matrices
    lora_alpha=16,  # The LoRA alpha parameter
    target_modules=["q_proj", "v_proj"],  # The names of the layers to apply LoRA
    lora_dropout=0.1,  # Dropout probability for LoRA
    bias="none",  # Bias handling in LoRA
    task_type="CAUSAL_LM",  # The type of task being fine-tuned
)

model = get_peft_model(model, lora_config)

# Step 2: Load and Preprocess the Dataset
try:
    with open('indicqa.hi.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Dataset file not found.")
    exit(1)

# Flatten the dataset structure and preprocess
contexts, questions, answers = [], [], []
for entry in data['data']:
    for paragraph in entry['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            contexts.append(context)
            questions.append(qa['question'])
            answers.append(qa['answers'][0]['text'] if qa['answers'] else "No answer provided")

# Define the preprocessing function
def preprocess_function(examples):
    inputs = [f"नीचे दिए गए संदर्भ को देखते हुए, प्रासंगिक प्रश्न-उत्तर युग्म उत्पन्न करें:\nसंदर्भ: {context}" for context in examples['context']]
    outputs = [f"प्रश्न: {question}\nउत्तर: {answer}" for question, answer in zip(examples['question'], examples['answer'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Prepare the dataset
dataset = Dataset.from_dict({"context": contexts, "question": questions, "answer": answers})
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 3: Set up Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset, 
)

# Step 4: Train and Save
trainer.train()
model.save_pretrained('./quantized_fine_tuned_llama')
tokenizer.save_pretrained('./quantized_fine_tuned_llama')
print("Fine-tuning complete! Model saved to './quantized_fine_tuned_llama'.")
