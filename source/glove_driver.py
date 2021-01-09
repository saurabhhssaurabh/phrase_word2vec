import os
import logging
import time
import pandas as pd

from phrase_glove import PhraseGlove

root_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "model")
tag_table_file = os.path.join(data_dir, "paper_tag_table.csv")
paper_table_file = os.path.join(data_dir, "paper_table.csv")
glove_file = os.path.join(model_dir, "glove.6B.50d.txt")

start_time = time.time()
tag_set = set(pd.read_csv(tag_table_file)['tag_text'].dropna().tolist())
abstract_list = pd.read_csv(paper_table_file)['abstract'].dropna().tolist()[:1]

obj = PhraseGlove(pre_trained_glove_file=glove_file)
obj.train(tag_set=tag_set, sentence_list=abstract_list)
end_time = time.time()
print(f"total time taken: {end_time - start_time}")