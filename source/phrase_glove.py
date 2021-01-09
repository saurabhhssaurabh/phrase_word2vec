from flashtext import KeywordProcessor
from sklearn.feature_extraction.text import CountVectorizer
from mittens.tf_mittens import Mittens

import logging
import gensim
import csv
import numpy as np
# import sys

# csv.field_size_limit(sys.maxsize)
logging.getLogger().setLevel(logging.DEBUG)

class PhraseGlove(object):

    def __init__(self, pre_trained_glove_file, tag_set, output_file, dimension=50, epochs=5):
        self.dimension = dimension
        self.epochs = epochs
        self.output_file = output_file

        self.kp = KeywordProcessor()
        self.pre_trained_glove_file = pre_trained_glove_file
        self.pre_trained_glove_embeds = self.load_glove_file(pre_trained_glove_file)
        self.create_kw_processor(tag_set)

    def create_output_glove_file(self):
        #writes new embeddings to file.
        with open(self.output_file, 'x', newline='') as embed_file:
            out_writer = csv.writer(embed_file, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            for key in self.pre_trained_glove_embeds.keys():
                out_writer.writerow([key] + list(self.pre_trained_glove_embeds[key]))
        logging.info("embeddings have been written to file: {self.output_file}")

    def create_kw_processor(self, tag_set):
        tag_phrase_dict = {}
        for tag in tag_set:
            index = tag.find('(')
            if index > -1:
                tag = tag[:index]
            tag = tag.strip().lower()
            tag_phrase_dict[tag.replace(' ', '_')] = [tag]

        self.kp.add_keywords_from_dict(tag_phrase_dict)

    def load_glove_file(self, glove_file):
        """
            Read word vectors into a dictionary.

            Arguments:
            glove_file: str, path to input glove file.

            Returns:
            embed:  dict, word embedding dictionary.
        """
        logging.info("generating word vector dictionary")
        with open(glove_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            embed = {line[0]: np.array(list(map(float, line[1:])))
                    for line in reader}
        return embed

    def pre_process(self, sentence_list):

        """
            Pre process input sentences.

            Arguments:
            sentence_list:  list, list of abstract of papers

            Returns:
            processed_sent_list:    list, list of processed sentences
        """

        logging.info('pre processing input sentences ')
        word_list = []
        for sent in sentence_list:
            tmp_sent = self.kp.replace_keywords(sent.strip().lower())
            for word in gensim.utils.tokenize(tmp_sent):
                word_list.append(word)
        
        logging.info(f"number of unique words {len(set(word_list))}")
        return word_list

    def train(self, sentence_list):
            """
            Main function to call to train word2vec on additional dataset.

            Arguments:
            tag_set:             set, set of keyword phrases
            sentence_list:       list, list of abstract of papers
            """
            word_list = self.pre_process(sentence_list)

            logging.info("generating cooccurrence matrix...")
            cv = CountVectorizer(ngram_range=(1,1))
            matrix = cv.fit_transform([' '.join(word_list)])
            cooccur_matrix = (matrix.T * matrix)
            cooccur_matrix.setdiag(0)
            cooccur_matrix = cooccur_matrix.toarray()
            logging.info(f"total number of words in co-occurrence matrix: {len(cooccur_matrix)}")
            logging.info(f"vocab of co-occurrence matrix: ")
            logging.info(cv.vocabulary_)
            
            #starts training
            logging.info("starting training...")
            mittens_model = Mittens(n=self.dimension, max_iter=self.epochs)
            new_embeddings = mittens_model.fit(cooccur_matrix,
                initial_embedding_dict= self.pre_trained_glove_embeds)
            logging.info(f"total number of words in new_embeddings dictionary: {len(new_embeddings)}")
            tmp_dict = cv.vocabulary_.copy()
            for key in tmp_dict.keys():
                tmp_dict[key] = new_embeddings[tmp_dict[key]]
            logging.info(f"updating embeddings with new embeddings...")
            self.pre_trained_glove_embeds.update(tmp_dict)