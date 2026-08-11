import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import argparse
import os
import pandas as pd

# 1. Update this to the local path where your CSV is stored inside your Jupyter environment
csv_path = 'newsCorpora.csv' 

df = pd.read_csv(csv_path, sep='\t', names=['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP'])
df = df[['TITLE','CATEGORY']]

my_dict = {
    'e':'Entertainment',
    'b':'Business',
    't':'Science',
    'm':'Health'
}

def update_cat(x):
    return my_dict.get(x, x)

df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))
df = df.reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())
        
        # Call the tokenizer directly instead of using .encode_plus()
        inputs = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long) 
        }

    def __len__(self):
        return self.len

train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_dataset.reset_index(drop=True)

print(f"Full dataset: {df.shape}")
print(f"Train dataset: {train_dataset.shape}")
print(f"Test dataset: {test_dataset.shape}")

MAX_LEN = 512

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def calculate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training loss per {nb_tr_steps} steps: {loss_step}")
            print(f"Training Accuracy per {nb_tr_steps} steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch {epoch}: {epoch_loss}")
    print(f"Training accuracy Epoch {epoch}: {epoch_accu}")

def valid(epoch, model, testing_loader, device, loss_function):
    model.eval()
    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 1000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation loss per {nb_tr_steps} steps: {loss_step}")
                print(f"Validation accuracy per {nb_tr_steps} steps: {accu_step}")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
    print(f"Validation accuracy epoch: {epoch_accu} at epoch {epoch}")

def main():
    print("Starting local training...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=16) # Increased default for 20GB GPU
    parser.add_argument("--valid_batch_size", type=int, default=8)  # Increased default for 20GB GPU
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()

    # Dataloaders mapped to arguments
    train_parameters = {
        'batch_size': args.train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    test_parameters = {
        'batch_size': args.valid_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_parameters)
    testing_loader = DataLoader(testing_set, **test_parameters)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f"\n--- Starting epoch: {epoch} ---")
        train(epoch, model, device, training_loader, optimizer, loss_function)
        valid(epoch, model, testing_loader, device, loss_function)

    # 2. Replaced SageMaker output directory with a local directory
    output_dir = './saved_model'
    os.makedirs(output_dir, exist_ok=True)

    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')

    torch.save(model.state_dict(), output_model_file)
    tokenizer.save_vocabulary(output_dir) # Note: save_vocabulary expects a directory
    print(f"\nTraining complete. Model saved locally to {output_dir}")

if __name__ == '__main__':
    main()