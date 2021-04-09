
from .vpcfg.model import *

model_factory = {
    'VGCPCFGs_MMT': VGCPCFGs_MMT,
    'VGCPCFGs': VGCPCFGs,
    'CPCFGs': CPCFGs,
    'Random': Random,
    'LeftBranching': LeftBranching,
    'RightBranching': RightBranching,
}

def get_model(name):
    return model_factory[name]