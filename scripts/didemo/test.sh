echo 'LeftBranching'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/LeftBranching.yaml  --test_mode --split test
echo 'RightBranching'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/RightBranching.yaml  --test_mode --split test
echo 'Random'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml  --test_mode --split test --seed 1110
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml  --test_mode --split test --seed 1111
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml  --test_mode --split test --seed 1112
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/Random.yaml  --test_mode --split test --seed 1113
echo 'C-PCFG'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/CPCFGs/CPCFGs_2020-11-15-14-33_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/CPCFGs/CPCFGs_2020-11-15-15-19_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/CPCFGs/CPCFGs_2020-11-15-16-05_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/CPCFGs.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/CPCFGs/CPCFGs_2020-11-15-16-52_1113.pkl
echo 'ResNeXt'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-15-19-00_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-15-18-11_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-15-18-08_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-resnext101-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-15-18-07_1113.pkl
echo 'SENet'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-15-21-03_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-15-20-12_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-15-20-09_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-senet154-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-15-20-07_1113.pkl
echo 'I3D'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-15-23-03_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-15-22-15_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-15-22-11_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-i3d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-15-22-08_1113.pkl
echo 'R2P1D'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-16-01-04_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-16-00-18_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-16-00-11_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-r2p1d-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-16-00-10_1113.pkl
echo 'S3DG'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-15-17-37_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-15-19-38_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-15-21-40_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-s3dg-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-15-23-40_1113.pkl
echo 'Scene'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-15-16-58_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-15-16-09_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-15-16-07_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-densenet161-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-15-16-06_1113.pkl
echo 'Audio'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-16-12-48_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-16-12-49_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-16-12-48_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-audio.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-16-12-48_1113.pkl
echo 'OCR'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-16-11-37_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-16-05-36_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-16-07-37_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-ocr.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-16-09-38_1113.pkl
echo 'Face'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-face/VGCPCFGs-face_2020-11-15-19-35_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-face/VGCPCFGs-face_2020-11-15-11-36_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-face/VGCPCFGs-face_2020-11-15-13-37_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-face.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-face/VGCPCFGs-face_2020-11-15-15-37_1113.pkl
echo 'Speech'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-16-03-36_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-15-23-35_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-16-01-37_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-speech.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-15-21-33_1113.pkl
echo 'Concat'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-15-14-58_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-15-14-07_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-15-14-05_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-Concat-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-15-14-04_1113.pkl
echo 'MMC-PCFG'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits_2020-11-15-12-27_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits_2020-11-15-11-35_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits_2020-11-15-11-35_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/DiDeMo/VGCPCFGs-MMT-avg-logits.yaml  --test_mode --split test --checkpoint final_checkpoints/DiDeMo/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits_2020-11-16-16-17_1113.pkl
