from helper import get_text_dataset,get_one_hot_dataset
import numpy as np
import os

texts, max_seq_len, tokens, num_tokens, token_index = get_text_dataset("preprocessed_corpus_file.txt")

texts = texts[2000:3000]

print('Number of samples:', len(texts))
print('Number of unique tokens:', num_tokens)
print('Max sequence length:', max_seq_len)

print(token_index)

input_data, target_data = get_one_hot_dataset(texts, max_seq_len, tokens, num_tokens, token_index)

batch_size = 128  # batch size for training
epochs = 40  # number of epochs to train for

from enc_dec_gan_model import create_model

model,enc,dec = create_model(num_tokens,pre_trained_weight=None)

model.fit([input_data, input_data], target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.08)
model.save("trained_weights\enc_dec_gan_weights.h5")