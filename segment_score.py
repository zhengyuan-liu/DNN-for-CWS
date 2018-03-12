from dl_for_cws import *

if __name__ == '__main__':

    # test_result_file = open('pku_test_result1.utf8', 'r', encoding='utf8')  # result of nn1
    test_result_file = open('pku_test_result2.utf8', 'r', encoding='utf8')  # result of nn2
    test_gold_file = open('corpus\pku_test_gold.utf8', 'r', encoding='utf8')
    result_all_lines = test_result_file.readlines()
    gold_all_lines = test_gold_file.readlines()
    result_words = 0  # number of all segmented words
    gold_words = 0    # number of all words in gold file
    right_words = 0   # number of correct segmented words
    for result_line, gold_line in zip(result_all_lines, gold_all_lines):
        result_words += len(result_line.split())
        gold_words += len(gold_line.split())
        result_tag_seq = Sentence(result_line).gold_tag_sequence
        gold_tag_seq = Sentence(gold_line).gold_tag_sequence
        head = True
        for tr, tg in zip(result_tag_seq, gold_tag_seq):
            if tr != tg:
                head = False
            elif tr == 0 and tg == 0:
                head = True
            elif tr == 2 and tg == 2 and head is True or tr == 3 and tg == 3:
                right_words += 1

    precision = right_words / result_words
    recall = right_words / gold_words
    F1 = 2 * precision * recall / (precision + recall)
    print('P:', precision)
    print('R:', recall)
    print('F:', F1)


