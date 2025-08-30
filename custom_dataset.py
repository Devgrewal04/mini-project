import torch
from transformers import BertTokenizer

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the input text
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove extra batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Same for attention mask
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are tensors
        }
