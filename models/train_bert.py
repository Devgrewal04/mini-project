from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

def train_bert(dataset: Dataset):  # Accept dataset directly
    # Initialize model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

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
        train_dataset=dataset
    )

    # Start training
    trainer.train()
