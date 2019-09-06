import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from helper import get_text_dataset,get_one_hot_dataset
import numpy as np

texts, max_seq_len, tokens, num_tokens, token_index = get_text_dataset("preprocessed_corpus_file.txt")
input_data, target_data = get_one_hot_dataset(texts, max_seq_len, tokens, num_tokens, token_index)

reverse_token_index = dict(
    (i, char) for char, i in token_index.items())

print(reverse_token_index)
# exit()

from enc_dec_gan_model import create_model
pre_trained_weight = "trained_weights/enc_dec_gan_weights.h5"
_,encoder_model,decoder_model = create_model(num_tokens,pre_trained_weight=pre_trained_weight)
def decode_sequence(input_seq):
    # encode the input sequence to get the internal state vectors.
    states_value = encoder_model.predict(input_seq)
    # requred_output = states_value[0]*100
    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, num_tokens))
    target_data[0,0,token_index['\t']] = 1
    # output sequence loop
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # sample a token and add the corresponding character to the 
        # decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_token_index[sampled_token_index]
        decoded_sentence += sampled_char
        # check for the exit condition: either hitting max length
        # or predicting the 'stop' character
        if (sampled_char == '\n' or 
            len(decoded_sentence) > max_seq_len):
            stop_condition = True
            
        # update the target sequence (length 1).
        target_seq = np.zeros((1, 1, num_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(10,11):
    input_seq = input_data[seq_index: seq_index + 1]
    print(input_seq)
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', texts[seq_index])
    print('Decoded sentence:', decoded_sentence)