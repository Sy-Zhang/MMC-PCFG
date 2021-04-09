import os
import argparse
import pprint
import pickle

from tqdm import tqdm
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

import _init_path
from core.config import update_config, cfg
from core.utils import create_logger, AverageMeter
from dataset.dataset_factory import get_dataset
from model.model_factory import get_model
from eval import get_batch_stats, initialize_stats, update_stats, get_f1s, display_f1s

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--workers', type=int, default=None, help='number of workers')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    # training
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--no_save', default=False, action="store_true", help='don\'t save checkpoint')
    parser.add_argument('--data_root', help='data path', type=str)
    parser.add_argument('--model_dir', help='model path', type=str)
    parser.add_argument('--log_dir', help='log path', type=str)
    parser.add_argument('--max_epoch', type=int, help='max epoch')
    parser.add_argument('--resume', default=False, action="store_true", help='train with punctuation')
    # testing
    parser.add_argument('--checkpoint', help='data path', type=str)
    parser.add_argument('--display_by_length', default=False, action="store_true", help='don\'t save checkpoint')
    parser.add_argument('--save_results', default=False, action="store_true", help='save predictions')
    parser.add_argument('--test_mode', default=False, action="store_true", help='run test epoch only')
    parser.add_argument('--split', help='test split', type=str)
    args = parser.parse_args()

    cfg.RESUME = args.resume
    if args.data_root:
        cfg.DATASET.DATA_ROOT = args.data_root
    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
    if args.log_dir:
        cfg.LOG_DIR = args.log_dir
    if args.checkpoint:
        cfg.CHECKPOINT = args.checkpoint
    if args.max_epoch:
        cfg.OPTIM.MAX_EPOCH = args.max_epoch
    if args.workers is not None:
        cfg.DATALOADER.WORKERS = args.workers

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.autograd.set_detect_anomaly(True)

    return args

def collate_fn(batch):
    video_features = []
    for feats in zip(*[b['video_features'] for b in batch]):
        video_features.append(torch.from_numpy(np.stack(feats, axis=0)).float())
    captions = pad_sequence([b['caption'] for b in batch], batch_first=True)
    caption_lengths = torch.tensor([b['caption'].shape[0] for b in batch], dtype=torch.long)
    spans = [b['span'] for b in batch]
    labels = [b['label'] for b in batch]
    raw_captions = [b['raw_caption'] for b in batch]
    return video_features, captions, caption_lengths, spans, labels, raw_captions

def network(batch, model, optimizer=None):
    video_features, captions, caption_lengths, gold_spans, labels, raw_captions = batch
    argmax_spans, loss, ReconPPL, KL, log_PPLBound = model(captions, caption_lengths, *video_features)

    pred_spans = [[[a[0], a[1]]  for a in span] for span in argmax_spans]
    loss_value = loss.mean()
    ReconPPL = ReconPPL.mean()
    KL = KL.mean()
    log_PPLBound = log_PPLBound.mean()

    if model.training:
        optimizer.zero_grad()
        loss_value.backward()
        if cfg.OPTIM.GRAD_CLIP > 0:
            clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()

    with torch.no_grad():
        stats = get_batch_stats(caption_lengths.tolist(), pred_spans, gold_spans, labels)
    predictions = [{'caption': caption, 'span': span} for caption, span in zip(raw_captions, argmax_spans)]
    return loss_value.item(), stats, ReconPPL.item(), KL.item(), log_PPLBound.item(), predictions

def run_epoch(data_loader, model, optimizer=None):
    if args.verbose:
        data_loader = tqdm(data_loader, dynamic_ncols=True)

    loss_meter = AverageMeter()
    ReconPPL_meter = AverageMeter()
    KL_meter = AverageMeter()
    log_PPLBound_meter = AverageMeter()
    all_stats = initialize_stats()
    predictions = []
    for batch in data_loader:
        caption_lengths = batch[2]
        loss, stats, ReconPPL, KL, log_PPLBound, pred_spans = network(batch, model, optimizer)
        predictions.extend(pred_spans)
        loss_meter.update(loss, 1)
        ReconPPL_meter.update(ReconPPL, torch.sum(caption_lengths).item())
        KL_meter.update(KL, len(caption_lengths))
        log_PPLBound_meter.update(log_PPLBound, torch.sum(caption_lengths).item())
        update_stats(all_stats, stats)
    f1s = get_f1s(all_stats)
    info = {'loss': loss_meter.avg, 'f1s': f1s, 'ReconPPL': ReconPPL_meter.avg,
            'KL': KL_meter.avg, 'log_PPLBound': log_PPLBound_meter.avg}
    return info, predictions

def test(cfg):
    test_dataset = get_dataset(cfg.DATASET.NAME)(cfg.DATASET, args.split)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.DATALOADER.INFERENCE_BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.DATALOADER.WORKERS,
                             pin_memory=True,
                             collate_fn=collate_fn)

    cfg.MODEL.PARAMS.word2int = test_dataset.word2int
    model = get_model(cfg.MODEL.NAME)(cfg.MODEL.PARAMS)
    if cfg.MODEL.NAME not in ['Random', 'LeftBranching', 'RightBranching']:
        assert os.path.exists(cfg.CHECKPOINT), "checkpoint not exists"
        checkpoint = torch.load(cfg.CHECKPOINT)
        model.load_state_dict(checkpoint['model'])
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        test_info, predictions = run_epoch(test_loader, model)

    result = display_f1s(test_info['f1s'], 'performance on testing set', args.display_by_length)
    print(result)
    if args.save_results:
        cfg_filename = os.path.splitext(os.path.basename(args.cfg))[0]
        saved_result_filename = os.path.join(cfg.RESULT_DIR, '{}/{}/{}-{}.pkl'.format(
            cfg.DATASET.NAME, cfg_filename, os.path.splitext(os.path.basename(args.checkpoint))[0], args.split))

        rootfolder = os.path.dirname(saved_result_filename)
        if not os.path.exists(rootfolder):
            print('Make directory %s ...' % rootfolder)
            os.makedirs(rootfolder)

        pickle.dump(predictions, open(saved_result_filename, 'wb'))


def train(cfg):
    logger, final_output_dir, log_filename = create_logger(cfg, args.cfg, args.tag)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n' + pprint.pformat(cfg))


    train_dataset = get_dataset(cfg.DATASET.NAME)(cfg.DATASET, 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.DATALOADER.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.DATALOADER.WORKERS,
                              pin_memory=True,
                              collate_fn=collate_fn)


    val_dataset = get_dataset(cfg.DATASET.NAME)(cfg.DATASET, 'val')
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.DATALOADER.INFERENCE_BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.DATALOADER.WORKERS,
                            pin_memory=True,
                            collate_fn=collate_fn)

    test_dataset = get_dataset(cfg.DATASET.NAME)(cfg.DATASET, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.DATALOADER.INFERENCE_BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.DATALOADER.WORKERS,
                             pin_memory=True,
                             collate_fn=collate_fn)

    cfg.MODEL.PARAMS.word2int = train_dataset.word2int
    model = get_model(cfg.MODEL.NAME)(cfg.MODEL.PARAMS)
    if os.path.exists(cfg.CHECKPOINT) and cfg.RESUME:
        logger.info("load model from {}".format(cfg.CHECKPOINT))
        checkpoint = torch.load(cfg.CHECKPOINT)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
    else:
        start_epoch = 0
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = getattr(optim, cfg.OPTIM.NAME)(model.parameters(), lr=cfg.OPTIM.LEARNING_RATE, betas=(cfg.OPTIM.BETA1, 0.999))

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        model.train()
        train_info, _ = run_epoch(train_loader, model, optimizer)

        model.eval()
        with torch.no_grad():
            val_info, _ = run_epoch(val_loader, model)
            test_info, _ = run_epoch(test_loader, model)

        optim_msg = "\nepoch: {} lr: {:.6f} ".format(cur_epoch, optimizer.param_groups[0]['lr'])
        train_display = display_f1s(train_info['f1s'],
                                    'Train [ReConPPL: {:.2f}, KL: {:.2f}, PPLBound: {:.2f}]'.format(
                                        train_info['ReconPPL'], train_info['KL'], np.exp(train_info['log_PPLBound'])),
                                    args.display_by_length)
        val_display = display_f1s(val_info['f1s'],
                                    'Val   [ReConPPL: {:.2f}, KL: {:.2f}, PPLBound: {:.2f}]'.format(
                                        val_info['ReconPPL'], val_info['KL'], np.exp(val_info['log_PPLBound'])),
                                  args.display_by_length)
        test_display = display_f1s(test_info['f1s'],
                                    'Test  [ReConPPL: {:.2f}, KL: {:.2f}, PPLBound: {:.2f}]'.format(
                                        test_info['ReconPPL'], test_info['KL'], np.exp(test_info['log_PPLBound'])),
                                   args.display_by_length)

        result_msg = train_display+'\n'+val_display+'\n'+test_display+'\n'
        logger.info(optim_msg+'\n'+result_msg+'\n')

        if not args.no_save:
            cfg_filename = os.path.splitext(os.path.basename(args.cfg))[0]
            saved_model_filename = os.path.join(cfg.MODEL_DIR, '{}/{}/{}.pkl'.format(
                cfg.DATASET.NAME, cfg_filename, os.path.splitext(log_filename)[0]))

            rootfolder = os.path.dirname(saved_model_filename)
            if not os.path.exists(rootfolder):
                print('Make directory %s ...' % rootfolder)
                os.makedirs(rootfolder)

            torch.save({'epoch': cur_epoch, 'model': model.module.state_dict()}, saved_model_filename)


if __name__ == '__main__':
    args = parse_args()
    if args.test_mode:
        test(cfg)
    else:
        train(cfg)