import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer


from torch import cuda
import os
device = 'cuda' if cuda.is_available() else 'cpu'
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
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

model = DistillBERTClass()
model.to(device)

def predict(model, tokenizer, input_text, max_len=512):
    model.eval()
    input_tokens = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True
    )
    input_ids = torch.tensor([input_tokens['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([input_tokens['attention_mask']], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax(output, dim=1)

    return prediction.item()


df = pd.read_csv(r'C:\Users\cadep\Desktop\Work\Dude-Deals\Modeling\Categories - All (2).csv', header=0)

df = df[['Name', 'Category', 'Dudedeals Category (Dudedeals Classification)']]
last_segment = df['Dudedeals Category (Dudedeals Classification)'].str.rsplit(';', n=1, expand=True)[1]

# Replace the original column with the last segment
df['Dudedeals Category (Dudedeals Classification)'] = last_segment

# Select the desired columns in the desired order
df = df[['Name', 'Category', 'Dudedeals Category (Dudedeals Classification)']]

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]
# Single text prediction
tokenizer = DistilBertTokenizer.from_pretrained(r'C:\Users\cadep\Desktop\Work\Dude-Deals\Modeling\.models\pytorch_distilbert_news.bin')
input_text = "Burton Screen Grab Liner Gloves"
predicted_category = predict(model, tokenizer, input_text)

# If you need to map the predicted category index back to its corresponding string label
predicted_label = list(encode_dict.keys())[list(encode_dict.values()).index(predicted_category)]
print(f"Predicted category index: {predicted_category}")
print(f"Predicted label: {predicted_label}")

# Full dataset prediction
new_dataset = ["Input text 1", "Input text 2", "Input text 3"]
predictions = []

for text in new_dataset:
    pred_category = predict(model, tokenizer, text)
    predictions.append(pred_category)

print(predictions)