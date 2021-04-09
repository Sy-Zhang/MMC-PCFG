from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle

UNK = 0

class PentathlonDataset(Dataset):
    def __init__(self, cfg, split):
        super(PentathlonDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.int2word, self.word2int = self.load_vocabularies()
        self.pairs = self.load_annotations()

        self.resnext101_features = self.load_segment_features(cfg.RESNEXT101_FEATURE_PATH) if 'resnext101' in cfg.EXPERTS else None
        self.senet154_features = self.load_segment_features(cfg.SENET154_FEATURE_PATH) if 'senet154' in cfg.EXPERTS else None
        self.i3d_features = self.load_segment_features(cfg.I3D_FEATURE_PATH) if 'i3d' in cfg.EXPERTS else None
        self.s3dg_features = self.load_segment_features(cfg.S3DG_FEATURE_PATH) if 's3dg' in cfg.EXPERTS else None
        self.r2p1d_features = self.load_segment_features(cfg.R2P1D_FEATURE_PATH) if 'r2p1d' in cfg.EXPERTS else None
        self.densenet161_features = self.load_segment_features(cfg.DENSENET161_FEATURE_PATH) if 'densenet161' in cfg.EXPERTS else None
        self.ocr_features = self.load_ocr_features() if 'ocr' in cfg.EXPERTS else None
        self.face_features = self.load_face_features() if 'face' in cfg.EXPERTS else None
        self.audio_features = self.load_audio_features() if 'audio' in cfg.EXPERTS else None
        self.speech_features = self.load_speech_features() if 'speech' in cfg.EXPERTS else None

    def load_annotations(self):
        pairs = []
        if self.split == 'train':
            subfolder = 'challenge-release-1'
            vids = [l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, subfolder, 'train_list.txt'))]\
                   +[l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, subfolder, 'val_list.txt'))]
        elif self.split == 'val':
            subfolder = 'challenge-release-1'
            vids = [l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, subfolder, 'public_server_val.txt'))]
        else:
            subfolder = 'challenge-release-2'
            vids = [l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, subfolder, 'public_server_test.txt'))]

        captions = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, subfolder, 'processed-captions.pkl'), 'rb'))
        sent2tree = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, subfolder, 'non_binary_tree.pkl'), 'rb'))

        for vid in vids:
            sentences = captions[vid]
            for tokens in sentences:
                if self.cfg.MAX_TEXT_LENGTH < 0 or len(tokens) < self.cfg.MAX_TEXT_LENGTH or self.split != 'train':
                    sent = ' '.join(tokens)
                    tree, span, label = sent2tree[sent]['tree'], sent2tree[sent]['span'], sent2tree[sent]['label']
                    pairs.append({'video_id': vid, 'sentence': sent, 'tree': tree, 'span': span, 'label': label})
        pairs = sorted(pairs, key=lambda x: len(x['sentence'].split(' ')))[::-1]
        return pairs

    def load_vocabularies(self):
        int2word = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.INT2WORD_PATH), 'rb'))
        word2int = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.WORD2INT_PATH), 'rb'))
        #add unknown
        int2word = {k+1:v for k, v in int2word.items()}
        word2int = {k:v+1 for k, v in word2int.items()}
        int2word[UNK] = '<unk>'
        word2int['<unk>'] = UNK

        return int2word, word2int


    def load_segment_features(self, path):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, path), 'rb'))
        if 'fixed_seg' in path:
            agg_features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, path.replace('fixed_seg', self.cfg.POOLING_TYPE)), 'rb'))
            for k, v in features.items():
                features[k] = np.concatenate([agg_features[k], features[k]], axis=0)
        return features

    def load_ocr_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.OCR_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,300), dtype=np.float32)
        return features

    def load_speech_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.SPEECH_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,300), dtype=np.float32)
        return features

    def load_audio_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.AUDIO_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,128), dtype=np.float32)
        return features

    def load_face_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.FACE_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,512), dtype=np.float32)
        return features

    def process_sentence(self, sentence):
        tokens = torch.tensor([self.word2int.get(word, UNK) for word in sentence.split(' ')], dtype=torch.long)
        return tokens

    def __getitem__(self, index):
        vid = os.path.splitext(self.pairs[index]['video_id'])[0]
        sent = self.pairs[index]['sentence']
        tree = self.pairs[index]['tree']
        span = self.pairs[index]['span']
        label = self.pairs[index]['label']

        features = []
        for key in self.cfg.EXPERTS:
            features.append(getattr(self, "{}_features".format(key))[vid])

        tokens = self.process_sentence(sent)
        item = {'video_features': features, 'caption': tokens, 'tree': tree, 'span': span, 'label': label, 'raw_caption': sent}
        return item

    def __len__(self):
        return len(self.pairs)