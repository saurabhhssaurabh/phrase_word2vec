import os
import pandas as pd
import csv
import numpy as np

root_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "model")
output_glove_file = os.path.join(model_dir, "output_glove.txt")
parent_tag_file = os.path.join(data_dir, "parent_tag.csv")
output_parent_tag_glove_file = os.path.join(model_dir, "parent_tag_output_glove.csv")

parent_tag_set = set(pd.read_csv(parent_tag_file)['parent_tag'].dropna().tolist())

parent_tag_list = []
for tag in parent_tag_set:
    index = tag.find('(')
    if index > -1:
        tag = tag[:index]
    tag = tag.strip().lower()
    index = tag.find('-')

    if index > -1:
        tag = tag.replace('-', ' ')
    
    parent_tag_list.append(tag.replace(' ', '_'))

with open(output_glove_file, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    embed = {line[0]: np.array(list(map(float, line[1:])))
            for line in reader}

parent_embed_dict = {}
for parent_tag in parent_tag_list:
    parent_embed_dict[parent_tag] = embed.get(parent_tag, None)

with open(output_parent_tag_glove_file, 'w+', newline='', encoding="utf-8") as embed_file:
    out_writer = csv.writer(embed_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for key in parent_embed_dict.keys():
        if parent_embed_dict.get(key) is not None:
            out_writer.writerow([key] + list(parent_embed_dict[key]))
        else:
            print(f"{key} => {str(None)}")