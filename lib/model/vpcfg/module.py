import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from core.config import cfg as config
from dataset.datasets.pentathlon_dataset import UNK

class ResLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x) + x

class CompoundCFG(nn.Module):
    def __init__(self, V, NT, T,
                 h_dim=512,
                 w_dim=512,
                 z_dim=64,
                 s_dim=256):
        super(CompoundCFG, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.root_emb = nn.Parameter(torch.randn(1, s_dim))
        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))

        self.rule_mlp = nn.Linear(s_dim + z_dim, self.NT_T ** 2)
        self.root_mlp = nn.Sequential(nn.Linear(s_dim + z_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        self.term_mlp = nn.Sequential(nn.Linear(s_dim + z_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        if z_dim > 0:
            self.enc_emb = nn.Embedding(V, w_dim)
            self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.enc_out = nn.Linear(h_dim * 2, z_dim * 2)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict)

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, l):
        x_embbed = self.enc_emb(x)
        self.enc_rnn.flatten_parameters()
        packed_x_embbed = pack_padded_sequence(x_embbed, l, batch_first=True, enforce_sorted=False)
        h, _ = self.enc_rnn(packed_x_embbed)
        unpacked_h = pad_packed_sequence(h, batch_first=True, padding_value=float('-inf'))[0]
        out = self.enc_out(unpacked_h.max(1)[0])

        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim:]
        return mean, lvar

    def forward(self, x, l):
        b, n = x.shape[:2]
        if self.z_dim > 0:
            mean, lvar = self.enc(x, l)
            kl = self.kl(mean, lvar).sum(1)
            z = mean
            if self.training: # NOTE: use mean value during evaluation
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob

        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            )
            if self.z_dim > 0:
                z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                    b, n, self.T, self.z_dim
                )
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            if self.z_dim > 0:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.no_imgnorm = cfg.no_imgnorm
        in_dim = sum([getattr(cfg, '{}_dim'.format(key)) for key in config.DATASET.EXPERTS])
        self.fc = nn.Linear(in_dim, cfg.sem_dim)

    def forward(self, *images):
        images = torch.cat(images, dim=-1).squeeze(1)
        features = self.fc(images)
        if not self.no_imgnorm:
            features = F.normalize(features, dim=-1)
        return features

class TextEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super(TextEncoder, self).__init__()
        self.NT = cfg.nt_states
        self.sem_dim = cfg.sem_dim
        self.syn_dim = cfg.syn_dim
        self.enc_rnn = torch.nn.LSTM(cfg.word_dim, cfg.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = torch.nn.Linear( cfg.lstm_dim * 2, self.NT * self.sem_dim)
        self.enc_emb = torch.nn.Embedding(len(cfg.word2int), cfg.word_dim, padding_idx=UNK)

    def _forward_srnn(self, x_emb, lengths):
        """
        lstm over every span, a.k.a. segmental rnn
        """
        b, N, dim = x_emb.size()
        word_mask = torch.arange(0, N, device=x_emb.device).unsqueeze(0).expand(b, N).long()
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        feats = torch.zeros(b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=x_emb.device)
        beg_idx = 0
        for k in range(1, N):
            inc = torch.arange(N - k, device=x_emb.device).view(N - k, 1)  # .expand(N - k, k + 1)
            idx = torch.arange(k + 1, device=x_emb.device).view(1, k + 1).repeat(N - k, 1)
            idx = (idx + inc).view(-1)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(b, -1, dim)

            feat = torch.gather(word_vect, 1, idx)
            feat = feat.view(b, N - k, k + 1, dim)
            feat = feat.view(-1, k + 1, dim)
            self.enc_rnn.flatten_parameters()
            feat = self.enc_out(self.enc_rnn(feat)[0])
            feat = feat.view(b, N - k, k + 1, self.NT, self.sem_dim)
            feat = F.normalize(feat.sum(2), dim=-1)
            end_idx = beg_idx + N - k
            feats[:, beg_idx: end_idx] = feat
            beg_idx = end_idx
        return feats

    def forward(self, captions, caption_lengths):
        word_emb = self.enc_emb(captions)
        return self._forward_srnn(word_emb, caption_lengths.cuda())


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = margin

    def forward(self, vid, txt):
        scores = vid.mm(txt.t()) # cosine similarity
        diagonal = scores.diag().view(vid.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(1)
        loss_img = loss_img.mean(0)
        return loss_txt + loss_img

class MixedContrastiveLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(MixedContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.gated_emb = GatedEmbedding(cfg)
        self.weight_predictor = nn.Sequential(nn.Linear(cfg.sem_dim, len(config.DATASET.EXPERTS)), nn.Softmax(dim=-1))
        self.margin = cfg.margin

    def forward(self, vid, txt):

        w = self.weight_predictor(txt)
        txt = self.gated_emb(txt)
        scores = torch.sum(w.t()[...,None]*vid.permute(1,0,2).bmm(txt.permute(1,2,0)), dim=0)
        diagonal = scores.diag().view(vid.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)

        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(1)
        loss_img = loss_img.mean(0)
        return loss_txt + loss_img, w

class GatedEmbedding(nn.Module):
    def __init__(self, cfg):
        super(GatedEmbedding, self).__init__()
        self.gated_embeddings = nn.ModuleList()
        for expert in config.DATASET.EXPERTS:
            self.gated_embeddings.append(nn.Linear(cfg.sem_dim, cfg.sem_dim))

    def forward(self, captions):
        outs = []
        for linear in self.gated_embeddings:
            z = linear(captions)
            z = z * torch.sigmoid(z)
            z = F.normalize(z, dim=-1)
            outs.append(z)
        outs = torch.stack(outs, dim=1)
        return outs

from .position_encoding import build_position_encoding
from .transformer import TransformerEncoder, TransformerEncoderLayer
class MultiModalTransformer(nn.Module):
    def __init__(self, cfg):
        super(MultiModalTransformer, self).__init__()
        self.cfg = cfg
        self.video_embeddings = nn.ModuleList()
        for expert in config.DATASET.EXPERTS:
            self.video_embeddings.append(nn.Linear(cfg.get("{}_dim".format(expert)), cfg.sem_dim))

        self.expert_embedding = nn.Embedding(len(config.DATASET.EXPERTS), cfg.sem_dim)
        self.position_embedding = build_position_encoding(cfg)

        encoder_layer = TransformerEncoderLayer(cfg.sem_dim, cfg.nhead, normalize_before=cfg.normalize_before)
        encoder_norm = nn.LayerNorm(cfg.sem_dim) if cfg.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, cfg.num_encoder_layers, encoder_norm)

    def forward(self, *videos):
        # videos listed as ['appearance', 'motion', 'audio', 'scene', 'ocr', 'face', 'speech']
        features = []
        expert_ids = []
        for i, (linear, feat) in enumerate(zip(self.video_embeddings, videos)):
            features.append(linear(feat))
            expert_ids.append(torch.full(feat.shape[:2], i, dtype=torch.long, device=feat.device))
        features = torch.cat(features, dim=1)
        expert_ids = torch.cat(expert_ids, dim=1)
        expert_embeddings = self.expert_embedding(expert_ids)

        position_embeddings = torch.cat([self.position_embedding(
            features, torch.zeros(v.shape[:2], dtype=torch.long, device=features.device)) for v in videos], dim=1)
        output = self.encoder(features.permute(1,0,2), pos=expert_embeddings.permute(1,0,2)+position_embeddings.permute(1,0,2))

        # Handle avg+fixed_seg
        if len(videos) != len(output):
            indexes = torch.cumsum(torch.tensor([0]+[v.shape[1] for v in videos[:-1]], dtype=torch.long, device=output.device), dim=0)
            output = output.index_select(0, indexes)
        output = output.permute(1, 0, 2)
        return output