import pickle
import os

if __name__ == '__main__':
    for root_dir in [
        'data/DiDeMo/challenge-release-1/',
        'data/YouCook2/challenge-release-1/',
        'data/MSRVTT/challenge-release-1/',
        'data/DiDeMo/challenge-release-2/',
        'data/YouCook2/challenge-release-2/',
        'data/MSRVTT/challenge-release-2/',
    ]:
        raw_captions = pickle.load(open(os.path.join(root_dir, 'raw-captions.pkl'), 'rb'))
        processed_captions = {}
        for vid, caption in raw_captions.items():
            for cap in caption:
                if 'wallcan' in cap:
                    idx = cap.index('wallcan')
                    cap[idx] = 'wall'
                    cap.insert(idx+1, 'can')
                if '.' in cap[-1]:
                    cap[-1] = cap[-1][:-1]
            processed_captions[vid] = caption
        pickle.dump(processed_captions, open(os.path.join(root_dir, 'processed-captions.pkl'), 'wb'))