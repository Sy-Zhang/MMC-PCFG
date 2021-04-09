CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/LeftBranching.yaml --verbose
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/RightBranching.yaml --verbose

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml --verbose --seed 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml --verbose --seed 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml --verbose --seed 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml --verbose --seed 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml --verbose --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml --verbose --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml --verbose --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml --verbose --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml --verbose  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml --verbose  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml --verbose  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml --verbose  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --seed 1113 --tag 1113

CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --seed 1110 --tag 1110
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --seed 1111 --tag 1111
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --seed 1112 --tag 1112
CUDA_VISIBLE_DEVICES=5 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --seed 1113 --tag 1113
