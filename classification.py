import os
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (RobertaForMultipleChoice,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments,
                          XLMRobertaForMultipleChoice, XLMRobertaTokenizer)

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
    print(type(torch.tensor(attention_masks).view(-1, 512)), torch.tensor(attention_masks).view(-1, 512).dtype)

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

    dataset = load_dataset(
        "csv", data_files={"train": args.train_path, "test": args.test_path}
    )

    model_name = "roberta-base"
    logging_dir = args.logging_dir

    print(f"Training {model_name} on {dataset}")

    metric = evaluate.load("accuracy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == "xlm-roberta-base":
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        print("Using the default roberta tokenizer, be careful")

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    if model_name == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
    elif model_name == "xlm-roberta-base":
        model = XLMRobertaForMultipleChoice.from_pretrained(model_name).to(device)
    else:
        model = RobertaForMultipleChoice.from_pretrained(model_name).to(device)
        print("Using the default roberta, be careful")

    output_dir = f"{logging_dir}/{model_name}"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
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
        report_to = "wandb"
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

    # Path where the checkpoints are saved
    checkpoints_path = output_dir
    checkpoints = [os.path.join(checkpoints_path, name) \
                    for name in os.listdir(checkpoints_path) \
                    if name.startswith("checkpoint")]

    # Placeholder for the best performance
    best_performance = 0.0
    best_checkpoint = None

    for checkpoint in checkpoints:
        # Load the model from checkpoint
        if model_name == "roberta-base":
            model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=4).to(device)
        elif model_name == "xlm-roberta-base":
            model = XLMRobertaForMultipleChoice.from_pretrained(checkpoint).to(device)
        else:
            model = RobertaForMultipleChoice.from_pretrained(checkpoint).to(device)
            print("Using the default roberta, be careful")

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=1,  # Adjust as necessary
            ),
            compute_metrics=compute_metrics,
        )

        # Evaluate the model
        eval_results = trainer.evaluate(tokenized_datasets['test'])

        # Assuming 'accuracy' is your metric of interest
        print(eval_results)
        performance = eval_results["eval_accuracy"]

        # Update the best checkpoint if current model is better
        if performance > best_performance:
            best_performance = performance
            best_checkpoint = checkpoint

    print(f"Best checkpoint: {best_checkpoint} with Eval Loss: {best_performance}")

    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} with Eval Loss: {best_performance}")

        # Load the best model
        if model_name == "roberta-base":
            best_model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=4).to(device)
        elif model_name == "xlm-roberta-base":
            best_model = XLMRobertaForMultipleChoice.from_pretrained(best_checkpoint).to(device)
        else:
            best_model = RobertaForMultipleChoice.from_pretrained(best_checkpoint).to(device)
            print("Using the default roberta, be careful")

        # Directly save the best model to the desired directory
        best_model.save_pretrained(f"{output_dir}/best_{best_checkpoint}")

        # If you want to save the tokenizer as well
        tokenizer.save_pretrained(f"{output_dir}/best_{best_checkpoint}")

        # Optional: Evaluate the best model again for confirmation, using the Trainer
        trainer = Trainer(
            model=best_model,
            args=TrainingArguments(
                output_dir=f'./{output_dir}/best',  # Ensure this matches where you're saving the model
                per_device_eval_batch_size=8,
            ),
            compute_metrics=compute_metrics,
        )

        eval_results = trainer.evaluate(tokenized_datasets['test'])
        print("Final Evaluation on Best Model:", eval_results)
    else:
        print("No best checkpoint identified.")





