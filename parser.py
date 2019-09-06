import os
import re

# A collection of multiple publically available books.
CORPUS_DIR = "./corpus"

corpuses = os.listdir(CORPUS_DIR)

def create_corpus():
    with open("text_corpus.txt","w") as db:
        for corpus in corpuses:
            path = os.path.join(CORPUS_DIR,corpus)
            book = open(path,"r")
            try:
                for line in book:
                    db.write(line)
            except:
                print("Error")
            book.close()
            print("Completed", corpus)

def preprocess_corpus():
    preprocess_corpus_file = open("preprocessed_corpus_file.txt","w")
    with open("text_corpus.txt","r") as corpus:
        corpus_s = corpus.read()
        corpus_s = re.sub(r"[\n]","",corpus_s)
        corpus_s = corpus_s.split(".")
        corpus_s = list(filter(None, corpus_s))
        for line in corpus_s:
            line = line.lower()
            line = re.sub(r"[^a-z0-9',. ]",'',line)
            if len(line) > 20:
                line = line.strip()
                preprocess_corpus_file.write("\t"+line+".\n\n")
preprocess_corpus()