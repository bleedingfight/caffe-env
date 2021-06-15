import caffe
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('./python')

sys.path.append('./examples/coco_caption')
vocabulary = ['<EOS>'] + [
    line.strip() for line in open(
        'examples/coco_caption/h5_data/buffer_100/vocabulary.txt').readlines()
]
net = caffe.Net('./examples/coco_caption/lstm_lm.deploy.prototxt',
                '/home/liushuai/下载/lstm.caffemodel', caffe.TEST)
print(net.blobs['probs'].data.shape)


def predict_single_word(net, previous_word, output='probs'):
    cont = 0 if previous_word == 0 else 1
    cont_input = np.array([cont])
    word_input = np.array([previous_word])
    net.forward(cont_sentence=cont_input, input_sentence=word_input)
    output_preds = net.blobs[output].data[0, 0, :]
    return output_preds


first_word_dist = predict_single_word(net, 0)
top_preds = np.argsort(-1 * first_word_dist)
print(top_preds[:10])
print([vocabulary[index] for index in top_preds[:10]])
second_word_dist = predict_single_word(net, vocabulary.index('two'))
print([vocabulary[index] for index in np.argsort(-1 * second_word_dist)[:10]])
third_word_dist = predict_single_word(net, vocabulary.index('giraffes'))
print([vocabulary[index] for index in np.argsort(-1 * second_word_dist)[:10]])
third_word_dist = predict_single_word(net, vocabulary.index('eating'))
print([vocabulary[index] for index in np.argsort(-1 * second_word_dist)[:10]])


def softmax(softmax_inputs, temp):
    shifted_inputs = softmax_inputs - softmax_inputs.max()
    exp_outputs = np.exp(temp * shifted_inputs)
    exp_outputs_sum = exp_outputs.sum()
    if np.isnan(exp_outputs_sum):
        return exp_outputs * float('nan')
    assert exp_outputs_sum > 0
    if np.isinf(exp_outputs_sum):
        return np.zeros_like(exp_outputs)
    eps_sum = 1e-20
    return exp_outputs / max(exp_outputs_sum, eps_sum)


def random_choice_from_probs(softmax_inputs, temp=1):
    # temperature of infinity == take the max
    if temp == float('inf'):
        return np.argmax(softmax_inputs)
    probs = softmax(softmax_inputs, temp)
    r = random.random()
    cum_sum = 0.
    for i, p in enumerate(probs):
        cum_sum += p
        if cum_sum >= r:
            return i
    return 1  # return UNK?


def generate_sentence(net, temp=float('inf'), output='predict', max_words=50):
    cont_input = np.array([0])
    word_input = np.array([0])
    sentence = []
    while len(sentence) < max_words and (not sentence or sentence[-1] != 0):
        net.forward(cont_sentence=cont_input, input_sentence=word_input)
        output_preds = net.blobs[output].data[0, 0, :]
        sentence.append(random_choice_from_probs(output_preds, temp=temp))
        cont_input[0] = 1
        word_input[0] = sentence[-1]
    return sentence


sentence = generate_sentence(net)
print(sentence)
print([vocabulary[index] for index in sentence])

sentence = generate_sentence(net)
print(sentence)
print([vocabulary[index] for index in sentence])

sentence = generate_sentence(net, temp=1.0)
print(sentence)
print([vocabulary[index] for index in sentence])

sentence = generate_sentence(net, temp=1.0)
print(sentence)
print([vocabulary[index] for index in sentence])
sentence = generate_sentence(net, temp=0.5)
print(sentence)
print([vocabulary[index] for index in sentence])
