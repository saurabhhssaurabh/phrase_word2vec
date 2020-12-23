from flashtext import KeywordProcessor
from sklearn.feature_extraction.text import CountVectorizer

import logging
import gensim
import csv
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

class PhraseGlove(object):

    def __init__(self, pre_trained_glove_file, dimension=50, epochs=5):
        self.dimension = 50
        self.epochs = epochs

        self.kp = KeywordProcessor()
        self.pre_trained_glove_file = pre_trained_glove_file
        self.pre_trained_glove_embeds = self.load_glove_file(pre_trained_glove_file)

    def load_glove_file(self, glove_file):
        with open(glove_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
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
            

        return word_list, list(oov_word_set)

def train(self, tag_set, sentence_list):
        """
           Main function to call to train word2vec on additional dataset.

           Arguments:
           tag_set:             set, set of keyword phrases
           sentence_list:       list, list of abstract of papers
           pre_trained_model:   string, path to pre_trained model
           output_model_file:   string, path to save final model.
        """
        word_list, oov_word_list = self.pre_process(tag_set, sentence_list)
        cv = CountVectorizer(ngram_range=(1,1), vocabulary=oov_word_list)
        X = cv.fit_transform([' '.join(word_list)])
        Xc = (X.T * X)
        Xc.setdiag(0)
        coocc_ar = Xc.toarray()


        self.model.build_vocab(train_data)
        self.model.intersect_word2vec_format(pre_trained_model, binary=True, lockf=1.0)
        logging.info("starting training...")
        self.model.train(train_data, epochs=5, total_examples=self.model.corpus_count)
        logging.info("training done...")
        logging.info(f'saving model at : {output_model_file}')
        self.model.wv.save_word2vec_format(output_model_file, binary=True)


