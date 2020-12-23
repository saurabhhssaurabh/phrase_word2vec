from flashtext import KeywordProcessor
from gensim.models import Word2Vec, KeyedVectors

import gensim
import logging

logging.getLogger().setLevel(logging.DEBUG)


class PhraseWord2VEC(object):

    def __init__(self, min_count_=1, size_=300, iter_=1):
        self.kp = KeywordProcessor()
        self.model = Word2Vec(min_count=min_count_, size=size_, iter=iter_)

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
        process_sent_list = []
        for sent in sentence_list:
            tmp_sent = self.kp.replace_keywords(sent.strip().lower())
            process_sent_list.append([word for word in gensim.utils.tokenize(tmp_sent)])

        return process_sent_list
        

    def train(self, tag_set, sentence_list, pre_trained_model, output_model_file):
        """
           Main function to call to train word2vec on additional dataset.

           Arguments:
           tag_set:             set, set of keyword phrases
           sentence_list:       list, list of abstract of papers
           pre_trained_model:   string, path to pre_trained model
           output_model_file:   string, path to save final model.
        """
        train_data = self.pre_process(tag_set, sentence_list)
        self.model.build_vocab(train_data)
        self.model.intersect_word2vec_format(pre_trained_model, binary=True, lockf=1.0)
        logging.info("starting training...")
        self.model.train(train_data, epochs=5, total_examples=self.model.corpus_count)
        logging.info("training done...")
        logging.info(f'saving model at : {output_model_file}')
        self.model.wv.save_word2vec_format(output_model_file, binary=True)


