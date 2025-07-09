import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def Average(lst): 
    return sum(lst) / len(lst) 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Path to the clipscore result file (JSONL format)')
    args = parser.parse_args()

    response = pd.read_json(path_or_buf=args.filename, lines=True)
    all_score = []

    for i in tqdm(range(len(response))):
        video_id = response.iloc[i]['videoid']
        scores = response.iloc[i]['response']
        score = [float(j) for j in scores]
        all_score.append(Average(score))

    print(f"Average CLIPScore across all videos: {Average(all_score):.4f}")
