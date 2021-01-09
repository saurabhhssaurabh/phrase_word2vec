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
glove_file = os.path.join(model_dir, "glove.txt")
output_glove_file = os.path.join(model_dir, "output_glove.txt")

start_time = time.time()
tag_set = set(pd.read_csv(tag_table_file)['tag_text'].dropna().tolist())
abstract_list = pd.read_csv(paper_table_file)['abstract'].dropna().tolist()[:1]

obj = PhraseGlove(pre_trained_glove_file=glove_file, tag_set=tag_set, output_file=output_glove_file)

start_index = 0
num_sents = 10000
if num_sents > len(abstract_list):
    end_index = len(abstract_list)
else:
    end_index = start_index + num_sents

while(end_index <= len(abstract_list)):
    sent_list = abstract_list[start_index:end_index]
    logging.info(f"sentences from index {start_index} to index {end_index}")
    obj.train(sentence_list=sent_list)

    start_index = end_index
    end_index = start_index + num_sents

logging.info("writing embeddings to output file...")
obj.create_output_glove_file()
print(f"total time taken: {time.time() - start_time}")