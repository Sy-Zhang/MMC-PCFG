CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/LeftBranching.yaml --verbose
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/RightBranching.yaml --verbose

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/CPCFGs-2k.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/CPCFGs-2k.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/CPCFGs-2k.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/CPCFGs-2k.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --verbose --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --verbose --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat.yaml --verbose  --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat.yaml --verbose  --seed 1115 --tag 1115

CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --verbose  --seed 1113 --tag 1113
CUDA_VISIBLE_DEVICES=6 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --verbose  --seed 1115 --tag 1115