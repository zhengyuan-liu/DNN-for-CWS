from dl_for_cws import *


# separate a sentence into characters
def separate_characters(str):
    ans = '  '
    for i in range(len(str)):
        ans += str[i]
        ans += '  '
    return ans

if __name__ == '__main__':

    corpora = [None] * 4
    corpora[0] = open('corpus/pku_training.utf8', 'r', encoding='utf8')
    corpora[1] = open('corpus/msr_training.utf8', 'r', encoding='utf8')
    corpora[2] = open('corpus/pku_test.utf8', 'r', encoding='utf8')
    corpora[3] = open('corpus/msr_test.utf8', 'r', encoding='utf8')
    unlabeled_corpus = open('corpus/unlabeled_corpus.utf8', 'w', encoding='utf8')
    for corpus in corpora:
        all_lines = corpus.readlines()
        for line in all_lines:
            sentence = Sentence(line)
            unlabeled_corpus.write(separate_characters(sentence.raw_sentence))
            unlabeled_corpus.write('\n')
        corpus.close()
    unlabeled_corpus.close()
