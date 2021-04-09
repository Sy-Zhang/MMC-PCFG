echo 'LeftBranching'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/LeftBranching.yaml --test_mode --split test
echo 'RightBranching'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/RightBranching.yaml --test_mode --split test
echo 'Random'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/Random.yaml --test_mode --split test --seed 1110
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/Random.yaml --test_mode --split test --seed 1111
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/Random.yaml --test_mode --split test --seed 1112
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/Random.yaml --test_mode --split test --seed 1113
echo 'C-PCFG'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/CPCFGs/CPCFGs_2020-11-18-16-17_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/CPCFGs/CPCFGs_2020-11-18-16-14_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/CPCFGs/CPCFGs_2020-11-18-16-17_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/CPCFGs.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/CPCFGs/CPCFGs_2020-11-18-22-22_1113.pkl
echo 'ResNeXt'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-18-11-32_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-18-22-57_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-18-11-05_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-resnext101-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-resnext101-avg-logits/VGCPCFGs-resnext101-avg-logits_2020-11-18-22-10_1113.pkl
echo 'SENet'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-19-10-20_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-19-21-37_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-19-09-32_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-senet154-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-senet154-avg-logits/VGCPCFGs-senet154-avg-logits_2020-11-19-20-50_1113.pkl
echo 'I3D'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-18-22-09_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-18-22-03_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-19-21-17_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-i3d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-i3d-avg-logits/VGCPCFGs-i3d-avg-logits_2020-11-18-21-58_1113.pkl
echo 'R2P1D'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-19-09-29_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-19-09-20_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-19-10-00_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-r2p1d-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-r2p1d-avg-logits/VGCPCFGs-r2p1d-avg-logits_2020-11-19-09-19_1113.pkl
echo 'S3DG'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-19-20-54_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-19-20-38_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-18-22-09_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-s3dg-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-s3dg-avg-logits/VGCPCFGs-s3dg-avg-logits_2020-11-19-20-36_1113.pkl
echo 'Scene'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-densenet161-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-18-23-17_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-densenet161-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-19-16-19_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-densenet161-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-18-23-17_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-densenet161-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-densenet161-avg-logits/VGCPCFGs-densenet161-avg-logits_2020-11-19-16-00_1113.pkl
echo 'Audio'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-19-04-21_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-18-16-08_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-19-07-33_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-audio.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-audio/VGCPCFGs-audio_2020-11-19-23-03_1113.pkl
echo 'OCR'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-20-00-49_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-18-16-08_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-19-07-36_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-ocr.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-ocr/VGCPCFGs-ocr_2020-11-19-23-00_1113.pkl
echo 'Face'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-18-16-08_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-18-11-59_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-19-03-15_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-19-18-32_1113.pkl
echo 'Speech'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-19-07-35_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-18-11-59_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-19-03-21_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-19-18-40_1113.pkl
echo 'Concat'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1113.pkl
echo 'MMC-PCFG'
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-18-04_1110.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-21-40_1111.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-21-40_1112.pkl
CUDA_VISIBLE_DEVICES=2 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-21-55_1113.pkl
