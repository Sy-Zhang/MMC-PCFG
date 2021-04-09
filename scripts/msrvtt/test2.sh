echo 'Face'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-18-16-08_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-18-11-59_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-19-03-15_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-face.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-face/VGCPCFGs-face_2020-11-19-18-32_1113.pkl
echo 'Speech'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-19-07-35_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-18-11-59_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-19-03-21_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-speech.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-speech/VGCPCFGs-speech_2020-11-19-18-40_1113.pkl
echo 'Concat'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-Concat-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-Concat-avg-logits/VGCPCFGs-Concat-avg-logits_2020-11-18-10-46_1113.pkl
echo 'MMC-PCFG'
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-18-04_1110.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-21-40_1111.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-21-40_1112.pkl
CUDA_VISIBLE_DEVICES=1 python parsing/run.py --cfg experiments/MSRVTT/VGCPCFGs-MMT-avg-logits.yaml --test_mode --split test --checkpoint final_checkpoints/MSRVTT/VGCPCFGs-MMT-avg-logits/VGCPCFGs-MMT-avg-logits-2k_2020-11-17-21-55_1113.pkl