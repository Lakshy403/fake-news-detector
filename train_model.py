from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = tokenizer(self.texts.iloc[idx], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        return {**encoding, 'labels': torch.tensor(self.labels.iloc[idx])}

def train_model(X_train, y_train):
    train_dataset = FakeNewsDataset(X_train, y_train)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8, save_steps=10, save_total_limit=2)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    
    trainer.train()
    model.save_pretrained("./fake_news_model")
    tokenizer.save_pretrained("./fake_news_model")
    
    return model
