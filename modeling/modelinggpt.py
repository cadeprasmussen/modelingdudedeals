key = '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'


import pandas as pd
import os
import json
import openai
import jsonlines

my_dict = []
df = pd.read_csv(r'C:\Users\Owner\Desktop\Work\Modeling\modeling\Categories - Sheet3 (1).csv')

for row in df.iterrows():
    desired = row[1]['Desired Category'].split('/')[-1].replace(';', '')
    string = {"prompt" :f"Return me a category from the following information: The name of the product and a given category. \n Name: {row[1]['Name']}\n Category:{row[1]['Category']}", "completion": desired}
    my_dict.append(string)


with jsonlines.open(os.path.realpath(os.path.join(os.path.dirname(__file__), "fine_tune.jsonl")), mode='w') as f:
    for dict in my_dict:
        f.write(dict)
# with open(r'C:\Users\Owner\Desktop\Work\Modeling\modeling\catList.txt', 'r') as f:
#     stop_cats = [line.strip() for line in f.readlines()]

# openai.api_key = key

# ft_model = "davinci:ft-dudedeals-2023-02-22-06-36-27"

# data = pd.read_json(r'C:\Users\Owner\Desktop\Work\Modeling\modeling\fine_tune_prepared_valid.jsonl', lines=True)

# data = "The North Face Boys' On Mountain 7 Inch Short - Medium - TNF Navy"
# res = openai.Completion.create(model=ft_model, prompt=data + " ->", max_tokens=10, temperature=.4, logprobs=2)
# print(res['choices'][0]['text'])

#Clothing;Clothing/Pants & Bottoms;Clothing/Pants & Bottoms/Sweats & Joggers;