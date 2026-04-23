# Install required libraries (run once)
# !pip install transformers datasets torch accelerate scikit-learn

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score

# ===============================
# 1. Load Dataset
# ===============================
dataset = load_dataset("emotion")

# ===============================
# 2. Load Model & Tokenizer
# ===============================
model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)

# ===============================
# 3. Device Setup (GPU/CPU)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===============================
# 4. Label Mapping
# ===============================
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# ===============================
# 5. Tokenization
# ===============================
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ===============================
# 6. Metrics
# ===============================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# ===============================
# 7. Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./emotion_model",
    evaluation_strategy="epoch",   # ✅ fixed
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,            # 🔥 better than 1
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available()
)

# ===============================
# 8. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),   # small for fast run
    eval_dataset=encoded_dataset["validation"].shuffle(seed=42).select(range(500)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ===============================
# 9. Train Model
# ===============================
trainer.train()

# ===============================
# 10. Save Model
# ===============================
model.save_pretrained("./final_emotion_model")
tokenizer.save_pretrained("./final_emotion_model")

# ===============================
# 11. Prediction Function
# ===============================
def predict_emotion_with_context(messages):
    combined_text = " ".join(messages)

    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return emotion_labels[prediction]

# ===============================
# 12. Chatbot Responses
# ===============================
emotion_responses = {
    "sadness": "I'm here for you. It's okay to feel sad sometimes.",
    "joy": "That's wonderful! I'm glad you're feeling happy 😊",
    "love": "That’s really beautiful ❤️",
    "anger": "I understand. Try taking a deep breath.",
    "fear": "You're not alone. Everything will be okay.",
    "surprise": "Oh! That sounds unexpected 😮"
}

# ===============================
# 13. Chatbot Loop
# ===============================
conversation_history = []

print("Emotion-Aware Chatbot (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    # store last 3 messages
    conversation_history.append(user_input)
    conversation_history = conversation_history[-3:]

    emotion = predict_emotion_with_context(conversation_history)
    response = emotion_responses[emotion]

    print("Detected Emotion:", emotion)
    print("Bot:", response, "\n")
