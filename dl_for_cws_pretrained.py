import numpy as np
import pickle

# Hyper-parameters
hyper_d = 50  # dimension of character embeddings
hyper_w = 5  # size of window
hyper_H = 300  # number of hidden units
hyper_lambda = 0.01  # learning rate

max_iterations = 8  # maximum iterations of training
attenuation_factor = 0.9  # adjust the learning rate after each epoch
corpus_size = 2000  # size of corpus


def get_random_vector(d):
    return -1 + 2 * np.random.random(d)


def get_random_matrix(row, col):
    return -1 + 2 * np.random.random((row, col))


class Sentence:
    @staticmethod
    def get_tag_seq(gold_sentence):
        tags = []
        words = gold_sentence.split()
        for word in words:
            if len(word) == 1:
                tags.append(3)  # 'S', a Single character
            else:
                tags.append(0)  # 'B', Begin of a word
                for i in range(2, len(word)):
                    tags.append(1)  # 'M', Middle of a word
                tags.append(2)  # 'E',  End of a word
        return tags

    def __init__(self, gold_sentence):
        self.gold_sentence = gold_sentence
        self.raw_sentence = gold_sentence.replace(' ', '')[0:-1]  # remove all the spaces and newline at the end of the sentence
        self.gold_tag_sequence = np.array(self.get_tag_seq(gold_sentence))
        self.length = len(self.gold_tag_sequence)


class NeuralNetwork:
    def __init__(self, sentences, dictionary, embedding_matrix):
        # Training data set
        self.corpus = sentences
        self.dictionary = dictionary  # map a character to an id
        self.D = len(self.dictionary)  # size of the dictionary

        # Hyper_parameters
        self.d = hyper_d  # dimension of character embeddings
        self.H = hyper_H  # number of hidden units
        self.window_size = hyper_w  # window size
        self.learning_rate = hyper_lambda  # learning rate
        self.tag_num = 4  # number of tags
        self.max_iterations = max_iterations  # maximum of iterations

        # Parameters to be trained
        self.M = embedding_matrix  # pre-trained character embedding matrix
        self.W1 = get_random_matrix(self.H, self.window_size * self.d)
        self.b1 = get_random_vector(self.H)
        self.W2 = get_random_matrix(self.tag_num, self.H)
        self.b2 = get_random_vector(self.tag_num)
        self.A0 = get_random_vector(self.tag_num)  # initial score for starting with the i-th tag
        self.A = get_random_matrix(self.tag_num, self.tag_num)  # transition score for jumping from i-th to j-th tag

        # Intermediate result
        self.hidden_layer_output = np.zeros(self.H)

        for i in range(self.max_iterations):
            error_count = self.perceptron_style_training()
            print('iteration: ', i, error_count, 1.0 - error_count/char_count)  # print approximate accuracy
            if error_count == 0:
                print('All correct!')
                break

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def get_tag_score(self, sentence, i):
        v = np.array([])
        k = int(self.window_size / 2)
        for j in range(-k, 0):
            if i + j < 0:
                char = 'begin'
            else:
                char = sentence[i + j]
            if char in self.dictionary:
                embedding = self.M[self.dictionary[char]]
            else:
                embedding = self.M[self.dictionary['unknown']]
            v = np.concatenate((v, embedding))
        for j in range(0, k + 1):
            if i + j >= len(sentence):
                char = 'end'
            else:
                char = sentence[i + j]
            if char in self.dictionary:
                embedding = self.M[self.dictionary[char]]
            else:
                embedding = self.M[self.dictionary['unknown']]
            v = np.concatenate((v, embedding))
        self.hidden_layer_output = self.sigmoid(np.dot(self.W1, v) + self.b1)
        tag_score = np.dot(self.W2, self.hidden_layer_output) + self.b2
        return tag_score

    def get_tag_score_matrix(self, sentence):
        tag_score_matrix = np.zeros((0, self.tag_num))
        for i in range(0, len(sentence)):
            tag_score = self.get_tag_score(sentence, i)
            tag_score_matrix = np.vstack((tag_score_matrix, tag_score))
        return tag_score_matrix

    def tag_inference(self, tag_score_matrix):
        length = tag_score_matrix.shape[0]
        max_score = np.zeros((length, self.tag_num))
        max_score_path = np.zeros((length, self.tag_num), dtype=np.int)
        for i in range(self.tag_num):
            max_score[0][i] = self.A0[i] + tag_score_matrix[0][i]
            max_score_path[0][i] = i
        for t in range(1, length):
            for i in range(self.tag_num):
                max_score_t = -10000
                backpointer = -1
                for j in range(self.tag_num):
                    score = max_score[t - 1][j] + self.A[j][i] + tag_score_matrix[t][i]
                    if score > max_score_t:
                        max_score_t = score
                        backpointer = j
                max_score[t][i] = max_score_t
                max_score_path[t][i] = backpointer
        tag_sequence = np.zeros(length, dtype=np.int)
        backpointer = np.argmax(max_score[-1])
        for t in range(length - 1, -1, -1):
            tag_sequence[t] = backpointer
            backpointer = max_score_path[t][backpointer]
        return tag_sequence

    def get_tag_sequence(self, sentence):
        if len(sentence) == 0:
            return []
        tag_score_matrix = self.get_tag_score_matrix(sentence)
        tag_sequence = self.tag_inference(tag_score_matrix)
        return tag_sequence

    def update(self, theta, pd_theta):
        return theta + self.learning_rate * pd_theta

    def perceptron_style_training(self):
        k = int(self.window_size / 2)
        error_count = 0
        count = 0
        for sentence in self.corpus:
            count += 1
            if count % 100 == 0:  # display training progress
                print(count)
            char_seq = sentence.raw_sentence
            if len(char_seq) == 0:
                continue
            gold_tag_seq = sentence.gold_tag_sequence
            predict_tag_seq = self.get_tag_sequence(sentence.raw_sentence)

            for (tg, tp) in zip(gold_tag_seq, predict_tag_seq):
                if tg != tp:
                    error_count += 1

            # initial the partial derivative of A0, A, f(t_i), f(t_i|c_i)
            pd_A0 = np.zeros(self.A0.shape)  # partial derivative of A0
            pd_A = np.zeros(self.A.shape)  # partial derivative of A
            pd_f_t = np.zeros(self.tag_num)  # partial derivative of f(t_i)
            pd_f_c = np.zeros((self.window_size, self.D, self.tag_num))  # partial derivative of f(t_i|c_i)

            # compute the partial derivative of A0, A, f(t_i), f(t_i|c_i)
            for i in range(0, len(gold_tag_seq)):
                tg = gold_tag_seq[i]
                tp = predict_tag_seq[i]
                if tg != tp:
                    if i == 0:
                        pd_A0[tg] += 1
                        pd_A0[tp] -= 1
                    else:
                        pd_A[gold_tag_seq[i - 1]][tg] += 1
                        pd_A[predict_tag_seq[i - 1]][tp] -= 1
                    pd_f_t[tg] += 1
                    pd_f_t[tp] -= 1

                    for j in range(-k, k+1):
                        if i + j < 0:
                            pd_f_c[k + j][self.dictionary['begin']][tg] += 1
                            pd_f_c[k + j][self.dictionary['begin']][tp] -= 1
                        elif i + j >= len(gold_tag_seq):
                            pd_f_c[k + j][self.dictionary['end']][tg] += 1
                            pd_f_c[k + j][self.dictionary['end']][tp] -= 1
                        else:
                            pd_f_c[k + j][self.dictionary[char_seq[i + j]]][tg] += 1
                            pd_f_c[k + j][self.dictionary[char_seq[i + j]]][tp] -= 1

            # compute the partial derivative of W2, b2, b1
            pd_b2 = pd_f_t
            pd_W2 = np.dot(pd_f_t.reshape(self.tag_num, 1), self.hidden_layer_output.reshape(1, self.H))
            pd_b1 = np.dot(pd_f_t, self.W2) * self.hidden_layer_output * (1 - self.hidden_layer_output)

            # compute the partial derivative of W1
            pd_W1_frag = np.zeros((self.window_size, self.H, self.d))
            for j in range(self.window_size):
                for i in range(self.D):
                    embedding = self.M[i]
                    temp = np.dot(pd_f_c[j][i], self.W2) * self.hidden_layer_output * (1 - self.hidden_layer_output)
                    pd_W1_frag[j] += np.dot(temp.reshape(self.H, 1), embedding.reshape(1, self.d))
            pd_W1 = pd_W1_frag[0]
            for i in range(1, self.window_size):
                pd_W1 = np.hstack((pd_W1, pd_W1_frag[i]))

            # computer the partial derivative of M
            pd_M = np.zeros(self.M.shape)
            for i in range(self.D):
                for j in range(self.window_size):
                    temp = np.dot(pd_f_c[j][i], self.W2) * self.hidden_layer_output * (1 - self.hidden_layer_output)
                    pd_M[i] += np.dot(temp, self.W1[:, j * self.d: (j+1) * self.d])

            # update parameters
            self.A0 = self.update(self.A0, pd_A0)
            self.A = self.update(self.A, pd_A)
            self.b2 = self.update(self.b2, pd_b2)
            self.W2 = self.update(self.W2, pd_W2)
            self.b1 = self.update(self.b1, pd_b1)
            self.W1 = self.update(self.W1, pd_W1)
            self.M = self.update(self.M, pd_M)
        self.learning_rate *= attenuation_factor

        return error_count


def get_embedding_from_line(line):
    values = line.split()
    character = values[0]
    vector = np.zeros(hyper_d)
    for i in range(1, len(values)):
        vector[i-1] = np.float64(values[i])
    return character, vector

if __name__ == '__main__':

    # read training corpus
    sentences = []
    training_corpus = open('corpus/pku_training.utf8', 'r', encoding='utf8')
    all_lines = training_corpus.readlines()
    t = 0
    char_count = 0
    for line in all_lines:
        sentence = Sentence(line)
        sentences.append(sentence)
        char_count += sentence.length
        t += 1
        if t == corpus_size:
            break
    training_corpus.close()
    print(t, 'training data is read.')

    # read pre-trained character embeddings
    dictionary = {}  # map a character to an id
    embedding_matrix = np.zeros((0, hyper_d))  # pre-trained character embedding matrix
    word2vector_file = open('model/word2vector.vector', 'r', encoding='utf8')
    all_lines = word2vector_file.readlines()
    all_vectors = all_lines[1:-1]
    char_id = 0
    for line in all_vectors:
        char, embedding = get_embedding_from_line(line)
        dictionary[char] = char_id
        embedding_matrix = np.vstack((embedding_matrix, embedding))
        char_id += 1
    word2vector_file.close()
    print(char_id, 'pre-trained embeddings are read.')
    # special characters (symbols)
    dictionary['begin'] = char_id
    embedding_matrix = np.vstack((embedding_matrix, get_random_vector(hyper_d)))
    char_id += 1
    dictionary['end'] = char_id
    embedding_matrix = np.vstack((embedding_matrix, get_random_vector(hyper_d)))
    char_id += 1
    dictionary['unknown'] = char_id
    embedding_matrix = np.vstack((embedding_matrix, get_random_vector(hyper_d)))
    char_id += 1

    # Train the neural network
    neural_network = NeuralNetwork(sentences, dictionary, embedding_matrix)

    # save NN to file
    nn_file = open('model/nn2', 'wb')
    pickle.dump(neural_network, nn_file)
    nn_file.close()
