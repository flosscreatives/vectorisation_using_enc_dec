import numpy as np

def get_text_dataset(preprocessed_corpus_file):
    texts = []
    tokens = set()
    
    num_tokens = 0
    max_seq_len = 0

    with open(preprocessed_corpus_file,"r") as corpus:
        for line in corpus:
            # print({"char":line})
            # exit()
            if(len(line.strip()) > 0):
                texts.append(line)
                for char in list(line):
                    if char not in tokens:
                        tokens.add(char)
    num_tokens = len(tokens)
    max_seq_len = max([len(line) for line in texts])

    tokens = sorted(tokens)

    token_index = dict(
        [(char, i) for i,char in enumerate(tokens)]
    )

    return texts, max_seq_len, tokens, num_tokens, token_index

def get_one_hot_dataset(texts, max_seq_len, tokens, num_tokens, token_index):
    input_data = np.zeros((len(texts),max_seq_len,num_tokens),dtype=np.float32)
    target_data = np.zeros((len(texts),max_seq_len,num_tokens),dtype=np.float32)

    for i, line in enumerate(texts):
        for j, char in enumerate(line):
            input_data[i,j,token_index[char]] = 1.
            if j > 0:
                target_data[i,j-1,token_index[char]] = 1

    return input_data, target_data