import os
import random

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from transformers import (RobertaForMultipleChoice,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments, AutoTokenizer,
                          ElectraForSequenceClassification, ElectraTokenizer, MistralForSequenceClassification)

from args_parser import get_args


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using CUDA
    random.seed(seed_value)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_function(examples):
    texts, labels = examples["text"], examples["label"]

    # Tokenize premises and choices
    # Note that we provide both choices together as multiple_choices_inputs
    multiple_choices_inputs = []
    for text in texts:
        multiple_choices_inputs.append(tokenizer.encode_plus( \
            text, max_length=512, padding='max_length', \
            truncation=True))

    # RoBERTa expects a list of all first choices and a list of all second 
    # choices, hence we restructure the inputs
    input_ids = [x['input_ids'] for x in multiple_choices_inputs]
    attention_masks = [x['attention_mask'] for x in multiple_choices_inputs]

    labels = np.unique(labels, return_inverse=True)[1]
    print("Labels are: ", np.unique(labels, return_inverse=True)[0])

    # Restructure inputs to match the expected format for RobertaForMultipleChoice
    features = {
        'input_ids': torch.tensor(input_ids).view(-1, 512),
        'attention_mask': torch.tensor(attention_masks).view(-1, 512),
        'labels': torch.tensor(labels)
    }
    return features


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    set_seed(1) 

    args = get_args()

    dataset = load_dataset("csv", data_files=
            {"train": f"./data/train_{args.run_name}.csv",
             "test": f"./data/test_{args.run_name}.csv"
    })

    model_name = args.model_name
    logging_dir = f"./logs/{args.run_name}"

    print(f"Training {model_name} on {dataset}")

    metric = evaluate.load("accuracy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == "electra":
        tokenizer = ElectraTokenizer.from_pretrained("bhadresh-savani/electra-base-emotion")
    elif model_name == "mistral":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        print("Using the default roberta tokenizer, be careful")

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["index", "text", "label"])

    if model_name == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
    elif model_name == "electra":
        model = ElectraForSequenceClassification.from_pretrained(model_name).to(device)
    elif model_name == "mistral":
        model = MistralForSequenceClassification.from_pretrained(model_name).to(device)
    else:
        model = RobertaForMultipleChoice.from_pretrained(model_name).to(device)
        print("Using the default roberta, be careful")

    output_dir = f"{logging_dir}/{model_name}"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=logging_dir,
        num_train_epochs=2.5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_ratio=0.07, 
        weight_decay=0.01,
        learning_rate=1e-5, 
        logging_dir='./logs',
        logging_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        report_to="wandb",
        run_name=args.run_name,
        load_best_model_at_end=True
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_predictions = model(
        torch.tensor(tokenized_datasets['test']["input_ids"]).to(device), 
        torch.tensor(tokenized_datasets["test"]["attention_mask"].to(device))
    )
    test_predictions = np.argmax(test_predictions.logits, axis=1)
    # Plot confusion matrix
    cm = confusion_matrix(tokenized_datasets['test']["label"], test_predictions)
    class_names = ["focus", "telling", "probing", "general"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    # plt.show()
    plt.savefig("window_1_open_ai.png")
