import os
import logging
import time
import pandas as pd

from phrase_word2vec import PhraseWord2VEC

root_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "model")
tag_table_file = os.path.join(data_dir, "paper_tag_table.csv")
paper_table_file = os.path.join(data_dir, "paper_table.csv")
pre_trained_model = os.path.join(model_dir, "GoogleNews-vectors-negative300.bin")
output_model_file = os.path.join(model_dir, "model.bin")

start_time = time.time()
tag_set = set(pd.read_csv(tag_table_file)['tag_text'].dropna().tolist())
abstract_list = pd.read_csv(paper_table_file)['abstract'].dropna().tolist()

logging.info(f"total number of sentences: {len(abstract_list)}")

obj = PhraseWord2VEC(iter_=5)
obj.train(tag_set=tag_set, sentence_list=abstract_list, pre_trained_model=pre_trained_model, output_model_file=output_model_file)
end_time = time.time()
print(f"total time taken: {end_time - start_time}")