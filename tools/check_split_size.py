import pickle
import os

def check_didemo(data_root, split):

    pairs = []
    if split == 'train':
        subfolder = 'challenge-release-1'
        vids = [l.rstrip('\n') for l in open(os.path.join(data_root, subfolder, 'train_list.txt'))] \
               + [l.rstrip('\n') for l in open(os.path.join(data_root, subfolder, 'val_list.txt'))]
    elif split == 'val':
        subfolder = 'challenge-release-1'
        vids = [l.rstrip('\n') for l in open(os.path.join(data_root, subfolder, 'public_server_val.txt'))]
    else:
        subfolder = 'challenge-release-2'
        vids = [l.rstrip('\n') for l in open(os.path.join(data_root, subfolder, 'public_server_test.txt'))]

    captions = pickle.load(open(os.path.join(data_root, subfolder, 'processed-captions.pkl'), 'rb'))
    sent2tree = pickle.load(open(os.path.join(data_root, subfolder, 'non_binary_tree.pkl'), 'rb'))

    for vid in vids:
        sentences = captions[vid]
        for tokens in sentences:
            if len(tokens) < 20 or split != 'train':
                sent = ' '.join(tokens)
                tree, span, label = sent2tree[sent]['tree'], sent2tree[sent]['span'], sent2tree[sent]['label']
                pairs.append({'video_id': vid, 'sentence': sent, 'tree': tree, 'span': span, 'label': label})
    pairs = sorted(pairs, key=lambda x: len(x['sentence'].split(' ')))[::-1]
    return pairs

if __name__ == '__main__':
    data_root = "data/YouCook2"
    split = 'train'
    pairs = check_didemo(data_root, split)
    print(len(pairs))

