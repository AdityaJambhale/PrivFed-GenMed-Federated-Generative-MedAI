import flwr as fl
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import sys

notes_file = sys.argv[1]

with open(notes_file, "r") as f:
    my_notes = f.readlines()


# Dummy data for THIS client
with open("client1_notes.txt", "r") as f:
    my_notes = f.readlines()

# Load tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Dataset & dataloader
class MedicalDataset(Dataset):
    def __init__(self, notes, tokenizer):
        self.notes = notes
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.notes[idx],
            truncation=True,
            padding='max_length',
            max_length=50,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

dataset = MedicalDataset(my_notes, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flower Client
class GPT2Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        optimizer = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(1):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Local loss: {loss.item():.4f}")
        if config["round"] == config["num_rounds"]:
            model.save_pretrained("PrivFedGenMed_Final")
            tokenizer.save_pretrained("PrivFedGenMed_Final")

        return self.get_parameters(config), len(dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(dataset), {}

fl.client.start_numpy_client(server_address="localhost:8080", client=GPT2Client())
