import pickle
import os
from collections import Counter
import numpy as np

if __name__ == '__main__':
    for data_root in [
        'data/MSRVTT/challenge-release-1',
        'data/YouCook2/challenge-release-1',
        'data/DiDeMo/challenge-release-1',
        ]:

        captions = pickle.load(open(os.path.join(data_root, 'processed-captions.pkl'), 'rb'))
        train_vids = [line.rstrip('\n')for line in open(os.path.join(data_root, 'train_list.txt'))]\
                     +[line.rstrip('\n')for line in open(os.path.join(data_root, 'val_list.txt'))]

        counter = Counter()
        for vid, sentences in captions.items():
            if vid not in train_vids:
                continue
            for sent in sentences:
                for word in sent:
                    counter[word] += 1

        sorted_words = [i[0] for i in sorted(counter.items(), key=lambda x: x[1], reverse=True)][:2000]
        word2int = {w: i for i,w in enumerate(sorted_words)}
        int2word = {i: w for i,w in enumerate(sorted_words)}
        pickle.dump(int2word, open(os.path.join(data_root, 'processed-int2word-2k.pkl'), 'wb'))
        pickle.dump(word2int, open(os.path.join(data_root, 'processed-word2int-2k.pkl'), 'wb'))