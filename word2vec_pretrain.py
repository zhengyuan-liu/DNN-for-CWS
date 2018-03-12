from gensim.models import word2vec

hyper_d = 50  # dimension of character embeddings

if __name__ == '__main__':
    training_corpus = 'corpus/unlabeled_corpus.utf8'
    output_model = 'model/word2vector.model'
    output_format = 'model/word2vector.vector'
    model = word2vec.Word2Vec(word2vec.LineSentence(training_corpus), size=hyper_d, min_count=1)
    model.save(output_model)
    model.wv.save_word2vec_format(output_format)
