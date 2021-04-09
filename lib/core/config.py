import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.LOG_DIR = ''
cfg.MODEL_DIR = ''
cfg.RESULT_DIR = ''
cfg.CHECKPOINT = ''
cfg.RESUME = False

# CUDNN related params
cfg.CUDNN = edict()
cfg.CUDNN.BENCHMARK = False
cfg.CUDNN.DETERMINISTIC = True
cfg.CUDNN.ENABLED = True

# grounding model related params
cfg.MODEL = edict()
cfg.MODEL.NAME = ''
cfg.MODEL.PARAMS = edict()

# DATASET related params
cfg.DATASET = edict()
cfg.DATASET.DATA_ROOT = ''
cfg.DATASET.NAME = ''
cfg.DATASET.EXPERTS = []
cfg.DATASET.POOLING_TYPE = ''
cfg.DATASET.WORD2INT_PATH = "challenge-release-1/processed-word2int-2k.pkl"
cfg.DATASET.INT2WORD_PATH = "challenge-release-1/processed-int2word-2k.pkl"
cfg.DATASET.TEXT_FEATURE_PATH = ''
cfg.DATASET.RESNEXT101_FEATURE_PATH = ''
cfg.DATASET.SENET154_FEATURE_PATH = ''
cfg.DATASET.I3D_FEATURE_PATH = ''
cfg.DATASET.S3DG_FEATURE_PATH = ''
cfg.DATASET.R2P1D_FEATURE_PATH = ''
cfg.DATASET.DENSENET161_FEATURE_PATH = ''
cfg.DATASET.AUDIO_FEATURE_PATH = ''
cfg.DATASET.OCR_FEATURE_PATH = ''
cfg.DATASET.FACE_FEATURE_PATH = ''
cfg.DATASET.SPEECH_FEATURE_PATH = ''
cfg.DATASET.MAX_TEXT_LENGTH = 40

cfg.DATALOADER = edict()
cfg.DATALOADER.WORKERS = 0
cfg.DATALOADER.BATCH_SIZE = 1
cfg.DATALOADER.INFERENCE_BATCH_SIZE = 4

# OPTIM
cfg.OPTIM = edict()
cfg.OPTIM.NAME = ''
cfg.OPTIM.LEARNING_RATE = 0.001
cfg.OPTIM.BETA1 = 0.9
cfg.OPTIM.MAX_EPOCH = 10
cfg.OPTIM.GRAD_CLIP = -1
cfg.OPTIM.CONTINUE = False

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if 'PARAMS' in k:
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(cfg[k], v)
                else:
                    cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))