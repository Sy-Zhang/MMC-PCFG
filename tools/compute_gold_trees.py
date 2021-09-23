import pickle
import benepar
benepar.download('benepar_en2')
from tqdm import tqdm
import os

def extract_spans_and_labels(tree, idx=0):
    spans = [[idx, idx+len(tree.leaves())-1]]
    labels = [tree.label()]
    start_idx = idx
    for node in tree:
        if len(node.leaves()) > 1:
            node_span, node_label = extract_spans_and_labels(node,start_idx)
            spans.extend(node_span)
            labels.extend(node_label)
        start_idx += len(node.leaves())
    return spans, labels

if __name__ == '__main__':
    parser = benepar.Parser("benepar_en2")
    for root_dir in ["data/DiDeMo/challenge-release-1/",
                     "data/YouCook2/challenge-release-1/",
                     "data/MSRVTT/challenge-release-1/",
                     "data/DiDeMo/challenge-release-2/",
                     "data/YouCook2/challenge-release-2/",
                     "data/MSRVTT/challenge-release-2/",
                    ]:
        captions = pickle.load(open(os.path.join(root_dir,'processed-captions.pkl'), 'rb'))

        sent2tree = {}
        for vid, sentences in tqdm(captions.items()):
            gold_trees = list(parser.parse_sents(sentences))
            spans, labels = [], []
            for tree in gold_trees:
                span, label = extract_spans_and_labels(tree)
                spans.append(span)
                labels.append(label)
            for sent, tree, span, label in zip(sentences, gold_trees, spans, labels):
                sent2tree.update({' '.join(sent): {'tree': str(tree), 'span': span, 'label': label}})

        pickle.dump(sent2tree, open(os.path.join(root_dir,"non_binary_tree.pkl"), 'wb'))
