import pickle
from dl_for_cws import *


def get_segment_sentence(sentence, tag_sequence):
    segment_sentence = ''
    for i in range(len(tag_sequence)):
        if tag_sequence[i] == 0:    # 'B', Begin of a word
            segment_sentence += sentence[i]
        elif tag_sequence[i] == 1:  # 'M', Middle of a word
            segment_sentence += sentence[i]
        elif tag_sequence[i] == 2:  # 'E',  End of a word
            segment_sentence += sentence[i]
            segment_sentence += "  "
        else:                       # 'S', a Single character
            segment_sentence += sentence[i]
            segment_sentence += "  "
    return segment_sentence

if __name__ == '__main__':

    # load NN from file
    nn_file = open('model/nn1', 'rb')
    neural_network = pickle.load(nn_file)
    nn_file.close()

    # load test data and write segmentation result to file
    test_dataset = open('corpus/pku_test.utf8', 'r', encoding='utf8')
    test_result_file = open('pku_test_result1.utf8', 'w', encoding='utf8')
    all_lines = test_dataset.readlines()
    for line in all_lines:
        sentence = line[0:-1]  # remove newline at the end of the sentence
        tag_sequence = neural_network.get_tag_sequence(sentence)
        segment_sentence = get_segment_sentence(sentence, tag_sequence)
        test_result_file.write(segment_sentence)
        test_result_file.write('\n')
    test_dataset.close()
    test_result_file.close()
