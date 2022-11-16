from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
# import random_split
from torch.utils.data import random_split
from torch.utils.data import Dataset

model_name = "bert-base-cased"
config = BertConfig.from_pretrained(
    model_name,
    num_labels=3,
)
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-cased",
    do_lower_case=False,
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    config=config,
)

trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
total_params = 0
trainable_params = 0

for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

for layer in trainable_layers:
    for p in layer.parameters():
        p.requires_grad = True
        trainable_params += p.numel()

print(f"Total parameters count: {total_params}") # ~108M
print(f"Trainable parameters count: {trainable_params}") # ~7M

class DepressionData(Dataset):
    def __init__(self, df, tokenizer):
        self.data = pd.read_csv(df)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'token_type_ids': encoded['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

dataset = DepressionData("dep1_cleaned.csv", tokenizer)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
BATCH_SIZE = 32

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

import torch

# Move the model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to train mode (HuggingFace models load in eval mode)
model = model.train()
# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

EPOCHS = 10
LOGGING_INTERVAL = 200 # once every how many steps we run evaluation cycle and report metrics
EPSILON = 7.5
DELTA = 1 / len(train_dataloader)


import numpy as np
from tqdm.notebook import tqdm

def accuracy(preds, labels):
    return (preds == labels).mean()

# define evaluation cycle
def evaluate(model):    
    model.eval()

    loss_arr = []
    accuracy_arr = []
    
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():

            outputs = model(**batch)
            loss, logits = outputs[:2]
            
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = batch['labels'].detach().cpu().numpy()
            
            loss_arr.append(loss.item())
            accuracy_arr.append(accuracy(preds, labels))
    
    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr)


from opacus import PrivacyEngine

MAX_GRAD_NORM = 0.1

privacy_engine = PrivacyEngine()

model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    target_delta=DELTA,
    target_epsilon=EPSILON, 
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)



from opacus.utils.batch_memory_manager import BatchMemoryManager

for epoch in range(1, EPOCHS+1):
    losses = []


    for step, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = batch
        outputs = model(**inputs) # output = loss, logits, hidden_states, attentions

        loss = outputs[0]
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if step > 0 and step % LOGGING_INTERVAL == 0:
            train_loss = np.mean(losses)
            eps = privacy_engine.get_epsilon(DELTA)

            eval_loss, eval_accuracy = evaluate(model)

            print(
                f"Epoch: {epoch} | "
                f"Step: {step} | "
                f"Train loss: {train_loss:.3f} | "
                f"Eval loss: {eval_loss:.3f} | "
                f"Eval accuracy: {eval_accuracy:.3f} | "
                f"ɛ: {eps:.2f}"
            )