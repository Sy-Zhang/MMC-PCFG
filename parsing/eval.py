import re
from collections import defaultdict
from core.utils import AverageMeter
from terminaltables import AsciiTable

def initialize_stats():
    per_label_f1 = defaultdict(list)
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = AverageMeter(), [0., 0., 0.]
    return per_label_f1, by_length_f1, sent_f1, corpus_f1

def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn

def get_batch_stats(lengths, pred_spans, gold_spans, labels):

    per_label_f1, by_length_f1, sent_f1, corpus_f1 = initialize_stats()
    for max_len, pred_span, gold_span, label in zip(lengths, pred_spans, gold_spans, labels):
        pred_set = set((a[0], a[1]) for a in pred_span if a[0] != a[1] and a != [0, max_len-1])
        gold_set = set((a[0], a[1]) for a in gold_span if a[0] != a[1] and a != [0, max_len-1])
        if len(gold_set) == 0:
            continue
        tp, fp, fn = get_stats(pred_set, gold_set)
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn
        overlap = pred_set.intersection(gold_set)
        prec = float(len(overlap)) / (len(pred_set) + 1e-8)
        reca = float(len(overlap)) / (len(gold_set) + 1e-8)

        if len(gold_set) == 0:
            reca = 1.
            if len(pred_set) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.update(f1)

        for l, gs in zip(label, gold_span):
            if gs[0] == gs[1] or gs == [0, max_len - 1]:
                continue
            l = re.split("=|-", l)[0]
            per_label_f1.setdefault(l, [0., 0.])
            per_label_f1[l][0] += 1

            lspan = gs[1] - gs[0] + 1
            by_length_f1.setdefault(lspan, [0., 0.])
            by_length_f1[lspan][0] += 1

            if tuple(gs) in pred_set:
                per_label_f1[l][1] += 1
                by_length_f1[lspan][1] += 1

    return per_label_f1, by_length_f1, sent_f1, corpus_f1

def update_stats(all_stats, batch_stats):
    for k, v in batch_stats[0].items():
        if k not in all_stats[0]:
            all_stats[0][k] = [0, 0]
        all_stats[0][k][0] += v[0]; all_stats[0][k][1] += v[1]
    for k, v in batch_stats[1].items():
        if k not in all_stats[1]:
            all_stats[1][k] = [0, 0]
        all_stats[1][k][0] += v[0]; all_stats[1][k][1] += v[1]
    all_stats[2].update(batch_stats[2].avg, batch_stats[2].count)
    all_stats[3][0] += batch_stats[3][0]
    all_stats[3][1] += batch_stats[3][1]
    all_stats[3][2] += batch_stats[3][2]

def get_corpus_f1(stat):
    prec = stat[0] / (stat[0] + stat[1]) if stat[0] + stat[1] > 0 else 0.
    recall = stat[0] / (stat[0] + stat[2]) if stat[0] + stat[2] > 0 else 0.
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    return f1

def get_f1s(stats):
    per_label_stat, by_length_stat, sent_stat, corpus_stat = stats


    f1s = {'C-F1': get_corpus_f1(corpus_stat), 'S-F1': sent_stat.avg}
    for k in ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]:
        if k in per_label_stat:
            f1s[k] = per_label_stat[k][1] / per_label_stat[k][0]
        else:
            f1s[k] = 0
    for k in by_length_stat.keys():
        if by_length_stat[k][0] >= 5:
            f1s[k] = by_length_stat[k][1]/by_length_stat[k][0]
    return f1s

def display_f1s(f1s, title=None, display_by_length=False):
    if display_by_length:
        column_names = list(range(2,16))
    else:
        column_names = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP", "C-F1", "S-F1"]
    display_data = [column_names]
    display_data.append(['{:.02f}'.format(f1s[k]*100) for k in column_names])
    table = AsciiTable(display_data, title)
    for i in range(2):
        table.justify_columns[i] = 'center'
    return table.table


import pickle
from glob import glob
import os
import numpy as np
if __name__ == '__main__':

    folders = ['LeftBranching', 'RightBranching', 'Random', 'CPCFGs-2k',
              'VGCPCFGs-resnext101-avg-logits', 'VGCPCFGs-senet154-avg-logits',
              'VGCPCFGs-i3d-avg-logits', 'VGCPCFGs-r2p1d-avg-logits',
              'VGCPCFGs-s3dg-avg-logits', 'VGCPCFGs-densenet161-avg-logits',
              'VGCPCFGs-audio', 'VGCPCFGs-ocr', 'VGCPCFGs-face', 'VGCPCFGs-speech',
               'VGCPCFGs-Concat-avg-logits', 'VGCPCFGs-MMT-avg-logits']

    sent_f1_matrix, corpus_f1_matrix = np.zeros((len(folders), len(folders))), np.zeros((len(folders), len(folders)))
    for i, folder_1 in enumerate(folders):
        for j, folder_2 in enumerate(folders):
            # if i > j:
            #     continue
            sent_f1_list, corpus_f1_list = [], []
            for k, pred_path in enumerate(glob(os.path.join('results/DiDeMo', folder_1, '*'))):
                for l, gt_path in enumerate(glob(os.path.join("results/DiDeMo", folder_2, '*'))):
                    if i==j and k >= l:
                        continue
                    predictions = pickle.load(open(pred_path, 'rb'))
                    ground_truths = pickle.load(open(gt_path, 'rb'))

                    sent_f1, corpus_f1 = AverageMeter(), [0., 0., 0.]
                    for prediction, ground_truth  in zip(predictions, ground_truths):
                        max_len = len(prediction['caption'].split(' '))
                        pred_span = prediction['span']
                        gold_span = ground_truth['span']
                        assert ground_truth['caption'] == prediction['caption']
                        pred_set = set((a[0], a[1]) for a in pred_span if a[0] != a[1] or a == (0, max_len-1))
                        gold_set = set((a[0], a[1]) for a in gold_span if a[0] != a[1] or a == (0, max_len-1))
                        if len(gold_set) == 0:
                            continue
                        tp, fp, fn = get_stats(pred_set, gold_set)
                        corpus_f1[0] += tp
                        corpus_f1[1] += fp
                        corpus_f1[2] += fn
                        overlap = pred_set.intersection(gold_set)
                        prec = float(len(overlap)) / (len(pred_set) + 1e-8)
                        reca = float(len(overlap)) / (len(gold_set) + 1e-8)

                        if len(gold_set) == 0:
                            reca = 1.
                            if len(pred_set) == 0:
                                prec = 1.
                        f1 = 2 * prec * reca / (prec + reca + 1e-8)
                        sent_f1.update(f1)

                    sent_f1_list.append(sent_f1.avg)
                    corpus_f1_list.append(get_corpus_f1(corpus_f1))
            print(i,j)
            sent_f1_matrix[i,j] = np.mean(sent_f1_list)
            corpus_f1_matrix[i, j] = np.mean(corpus_f1_list)
    print(np.around(sent_f1_matrix*100, 2).tolist())
    print(np.around(corpus_f1_matrix*100, 2).tolist())