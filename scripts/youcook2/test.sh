echo 'LeftBranching'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/LeftBranching.yaml --test_mode --split test
echo 'RightBranching'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/RightBranching.yaml --test_mode --split test
echo 'Random'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --test_mode --split test --seed 1111
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --test_mode --split test --seed 1112
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --test_mode --split test --seed 1113
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/Random.yaml --test_mode --split test --seed 1115
echo 'C-PCFG'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/CPCFGs-2k/CPCFGs-2k_2020-11-01-23-41_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/CPCFGs-2k/CPCFGs-2k_2020-11-01-23-41_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/CPCFGs-2k/CPCFGs-2k_2020-11-01-23-41_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/CPCFGs-2k/CPCFGs-2k_2020-11-01-23-41_1115.pkl
echo 'ResNeXt'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-02-14-25_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-02-14-21_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-02-14-18_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-03-06-22_1115.pkl
echo 'SENet'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-02-16-45_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-02-16-38_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-02-16-33_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-03-08-39_1115.pkl
echo 'I3D'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-02-19-04_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-02-18-56_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-02-18-49_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-03-06-13_1115.pkl
echo 'R2P1D'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-02-21-22_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-02-21-13_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-02-21-06_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-03-08-29_1115.pkl
echo 'S3DG'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-02-21-38_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-02-22-14_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-02-21-52_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-02-16-42_1115.pkl
echo 'Audio'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-03-01-59_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-10-20-10_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-10-20-10_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-10-20-10_1115.pkl
echo 'OCR'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-02-21-18_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-02-21-19_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-02-21-18_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-02-19-39_1115.pkl
echo 'Concat'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-Concat/VGCPCFGs-Concat_2020-11-02-09-15_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-Concat/VGCPCFGs-Concat_2020-11-02-09-15_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-Concat/VGCPCFGs-Concat_2020-11-02-09-15_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-Concat/VGCPCFGs-Concat_2020-11-03-06-37_1115.pkl
echo 'MMC-PCFG'
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT_2020-11-02-11-36_1111.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT_2020-11-02-11-33_1112.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT_2020-11-02-11-31_1113.pkl
CUDA_VISIBLE_DEVICES=3 python parsing/run.py --cfg experiments/YouCook2/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/YouCook2/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT_2020-11-02-21-22_1115.pkl