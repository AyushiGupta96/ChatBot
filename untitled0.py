

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import gradio as gr

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained DialoGPT model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Baseline chatbot function
chat_history_ids = None

def chatbot_response(user_input, chat_history_ids=None):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    # Add conversational history
    bot_input_ids = torch.cat([chat_history_ids , new_input_ids], dim= -1)if chat_history_ids is not None else  new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

css = """
/* Container */
.container {
    background-color: #fdf4f4;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    padding: 25px;
    font-family: 'Comic Sans MS', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    font-size: 32px;
    color: #ff7f7f;
    font-weight: 600;
    margin-bottom: 25px;
    font-family: 'Pacifico', sans-serif;
}

/* Outer box */
.input_output_outerbox {
    background-color: #f8d3d3; /* Light pink */
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 15px;
}

/* Input and Text area */
input[type="text"], textarea {
    width: 100%;
    padding: 18px 22px;
    font-size: 18px;
    border-radius: 25px;
    border: 2px solid #ff6f61;
    background-color: #fff9e6; /* Cream color */
    color: brown;
    font-weight: bold;
    outline: none;
    transition: border-color 0.3s ease;
}

/* Keep background and text color on focus */
input[type="text"]:focus, textarea:focus {
    border-color: #ff1493;
    background-color: #fff9e6 !important;
    color: brown;
    font-weight: bold;
    box-shadow: none;
}

/* Output */
.output_text {
    padding: 16px 22px;
    background-color: #2e082e;
    border-radius: 20px;
    font-size: 18px;
    color: brown;
    font-weight: bold;
    border: 1px solid #ff6f61;
    word-wrap: break-word;
    min-height: 60px;
}

/* Button */
button {
    background-color: #ff6f61;
    color: red;
    padding: 16px 28px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 30px;
    border: none;
    cursor: pointer;
    width: 100%;
    transition: background-color 0.3s ease, transform 0.2s;
}

/* Button hover effect with animation */
button:hover {
    background-color: #ff1493;
    transform: scale(1.1);
}

/* Cute footer with smaller text */
footer {
    text-align: center;
    margin-top: 20px;
    font-size: 16px;
    color: #ff6f61;
}

"""

iface = gr.Interface(fn=chatbot_response,
                     theme="default",
                     inputs="text",
                     outputs="text",
                     title="Baseline Chatbot",
                     css=css)
iface.launch()

# Load a subset of DailyDialog
dataset = load_dataset("daily_dialog")
train_data = dataset["train"].shuffle(seed=42).select(range(len(dataset["train"]) // 10))
valid_data = dataset["validation"].shuffle(seed=42).select(range(len(dataset["validation"]) // 10))

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Flatten multi-turn dialog structure
    text_list = [" ".join(dialog) if isinstance(dialog , list) else dialog for dialog in examples["dialog"]]

    # Tokenize each conversation
    model_inputs = tokenizer(text_list, truncation=True, padding="max_length", max_length=128)

    # Set labels = input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

# Tokenizing dataset
tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["dialog"])
tokenized_valid = valid_data.map(tokenize_function, batched=True, remove_columns=["dialog"])

# Convert dataset format
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./fine_tuned_chatbot",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid
)

# Train the model
trainer.train()

def chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=30,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response



# Gradio UI
iface = gr.Interface(fn = chatbot_response, inputs="text",outputs = "text" , title = "Trained chatbot")
iface.launch()

# Small knowledge base
knowledge_base = {
    "huggingface": "Hugging Face is a company specializing in Natural Language Processing technologies.",
    "transformers": "Transformers are a type of deep learning model introduced in the paper 'Attention is All You Need'.",
    "gradio": "Gradio is a Python library that allows you to rapidly create user interfaces for machine learning models."
}

def retrieve_relevant_info(query):
    # Simple keyword matching
    for keyword, info in knowledge_base.items():
        if keyword.lower() in query.lower():
            return info
    return ""

def chatbot_response(user_input):
    retrieved_info = retrieve_relevant_info(user_input)
    augmented_prompt = (retrieved_info + "\n" if retrieved_info else "") + "User: " + user_input + "\nBot:"

    input_ids = tokenizer.encode(augmented_prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.85,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

conversation_history = []

def chatbot_response(user_input):
    global conversation_history
    conversation_history.append(f"User: {user_input}")
    if len(conversation_history) > 6:  # Limit to last 6 turns
        conversation_history = conversation_history[-6:]

    prompt = "\n".join(conversation_history) + "\nBot:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.85,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

    conversation_history.append(f"Bot: {response}")
    return response

conversation_history = []

def chatbot_response(user_input):
    global conversation_history
    conversation_history.append(f"User: {user_input}")
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]

    prompt = "\n".join(conversation_history) + "\nBot:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        top_k=50,
        repetition_penalty=1.2
    )

    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

    # Fallback if response is too short or vague
    if not response or len(response.split()) <= 2:
        response = "I'm not sure I understood that. Could you please rephrase?"

    conversation_history.append(f"Bot: {response}")
    return response