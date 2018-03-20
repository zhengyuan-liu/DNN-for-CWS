# DNN-for-CWS
Deep neural network for Chinese word segmentation

Method Used see here https://github.com/zhengyuan-liu/DNN-for-CWS/blob/master/Reference-Paper-Summary.pdf

## Codes:
dl_for_cws.py: initialize and train a DNN for Chinese Word Segmentation on training data set (NN1)

dl_for_cws_pretrained.py: initialize a DNN by pre-trained character embeddings and train it on training data set (NN2)

nn1_test.py: generate segmentation results on test data using NN1

nn2_test.py: generate segmentation results on test data using NN2

segment_score.py: get precision P, recall R, and F1-score F for the segmentation task

build_unlabeled_corpus.py: build unlabeled corpus based on PKU and MSRA training data set

word2vec_pretrain.py: get pre-trained character embeddings by word2vec toolkit

## Models:
nn1: DNN trained by dl_for_cws.py

nn2: DNN trained by dl_for_cws_pretrained

word2vector.vector: pre-trained character embeddings by word2vec toolkit

## Data:
pku_training(.txt and .utf8): PKU training data set

pku_test(.txt and .utf8): PKU test data set

pku_test_gold(.txt and .utf8): gold segmentation of PKU test data set

msr_training(.txt and .utf8): MSRA training data set

msr_test(.txt and .utf8): MSRA training data set

unlabeled_corpus.utf8: unlabeled corpus to train the word2vec model

## Results:
pku_test_result1.utf8: segmentation result of PKU test data set (NN1)

pku_test_result2.utf8: segmentation result of PKU test data set (NN2)
