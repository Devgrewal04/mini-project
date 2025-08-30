# Assuming train_distilbert.py is in the models directory

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch

def train_distilbert(dataloader: DataLoader):  # Now it accepts the dataloader argument
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # Update num_labels based on your dataset

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./outputs',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataloader.dataset,  # Pass dataset, not DataLoader
    )

    # Start training
    trainer.train()
