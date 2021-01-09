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

    def __init__(self, pre_trained_glove_file, dimension=50, epochs=5):
        self.dimension = 50
        self.epochs = epochs

        self.kp = KeywordProcessor()
        self.pre_trained_glove_file = pre_trained_glove_file
        self.pre_trained_glove_embeds = self.load_glove_file(pre_trained_glove_file)

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

    def pre_process(self, tag_set, sentence_list):

        """
            Pre process input sentences.

            Arguments:
            tag_set:        set, set of keyword phrases
            sentence_list:  list, list of abstract of papers

            Returns:
            processed_sent_list:    list, list of processed sentences
        """
        tag_phrase_dict = {}
        for tag in tag_set:
            tag = tag.strip().lower()
            tag_phrase_dict[tag.replace(' ', '_')] = [tag]

        self.kp.add_keywords_from_dict(tag_phrase_dict)

        logging.info('pre processing input sentences ')
        word_list = []
        oov_word_set = set()
        for sent in sentence_list:
            tmp_sent = self.kp.replace_keywords(sent.strip().lower())
            for word in gensim.utils.tokenize(tmp_sent):
                word_list.append(word)
                if word not in self.pre_trained_glove_embeds.keys():
                    oov_word_set.add(word)
        
        logging.info(f"number of oov words {len(oov_word_set)}")
        return word_list, list(oov_word_set)

    def train(self, tag_set, sentence_list):
            """
            Main function to call to train word2vec on additional dataset.

            Arguments:
            tag_set:             set, set of keyword phrases
            sentence_list:       list, list of abstract of papers
            """
            word_list, oov_word_list = self.pre_process(tag_set, sentence_list)

            logging.info("generating cooccurrence matrix...")
            cv = CountVectorizer(ngram_range=(1,1), vocabulary=oov_word_list)
            matrix = cv.fit_transform([' '.join(word_list)])
            cooccur_matrix = (matrix.T * matrix)
            cooccur_matrix.setdiag(0)
            cooccur_matrix = cooccur_matrix.toarray()
            
            #starts training
            logging.info("starting training...")
            mittens_model = Mittens(n=50, max_iter=1000)
            new_embeddings = mittens_model.fit(cooccur_matrix, vocab=oov_word_list,
                initial_embedding_dict= self.pre_trained_glove_embeds)

            logging.info("writing new embedding to file...")
            #writes new embeddings to file.
            with open(self.pre_trained_glove_file, 'a', newline='') as embed_file:
                out_writer = csv.writer(embed_file, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(oov_word_list)):
                    out_writer.writerow([oov_word_list[i]] + list(new_embeddings[i]))