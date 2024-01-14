import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch 
from torch.utils.data import DataLoader, Dataset


class PlagiarismDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        try:
            label = int(self.labels[idx])
        except ValueError:
            # Handle the case where the label is not a valid integer
            print(f"Warning: Invalid label '{self.labels[idx]}' at index {idx}. Setting label to 0.")
            label = 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PlagiarismModel:
    def __init__(self, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device

    def train(self, train_texts, train_labels, epochs=3, batch_size=16, learning_rate=2e-5):
        train_dataset = PlagiarismDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            average_loss = total_loss / len(train_loader)
        
        
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

    def predict(self, texts):
        self.model.eval()
        encoded_texts = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits

        probabilities = torch.softmax(logits, dim=1)
        return probabilities[:, 1].cpu().numpy()

    def predict_with_similar_text(self, texts):
            self.model.eval()
            encoded_texts = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded_texts['input_ids'].to(self.device)
            attention_mask = encoded_texts['attention_mask'].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits

            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()

            # Extract similar text
            similar_texts = [text if label == 1 else "" for text, label in zip(texts, predicted_labels)]
            return probabilities[:, 1].cpu().numpy(), similar_texts

def train_plagiarism_model(train_texts, train_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    plagiarism_model = PlagiarismModel(tokenizer, model)

    if len(train_texts) == 0 or len(train_labels) == 0:
        print("Error: No training data provided.")
        return None

    plagiarism_model.train(train_texts, train_labels)

    return plagiarism_model
