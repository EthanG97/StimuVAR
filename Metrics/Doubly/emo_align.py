
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from difflib import SequenceMatcher

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--response_path', type=str,default="",  help="path of the output file")
    args = parser.parse_args()
    gptfile = args.response_path.split('.jsonl')[0] + '_gpt.jsonl'
    file_path = args.response_path
    response_con = 0
    rank1 = 0
    rank3 = 0
    rank3_g = 0
    RR = 0
    RW = 0
    WR = 0
    WW = 0
    response = pd.read_json(path_or_buf=file_path,lines=True)
    gptresponse = pd.read_json(path_or_buf=gptfile,lines=True)
    for i in tqdm(range(len(gptresponse))):
        video_id = response.iloc[i]['videoid']
        responses = response.iloc[i]['response'][:-1]
        gt = response.iloc[i]['gt']

        label, reason = responses.split("""'''\n""")
        filtered, reason = reason.split('because')
        gptresonses = gptresponse.iloc[i]['response']
        if 'because' not in gptresonses:
            continue
        gptlabel, gptreason = gptresonses.split('because', 1)
        gptlabel = gptlabel.split('The viewer feels')[-1]
        match = SequenceMatcher(None, gptlabel, filtered).find_longest_match()
        gpt_correct = 0
        pre_correct = 0
        if match.size > 3:
            response_con += 1
        for i,g in enumerate(gt):
            if i == 0 and g.lower() in gptlabel.lower():
                rank3_g += 1
                gpt_correct = 1
                break
            else:
                if g.lower() in gptlabel.lower():
                    gpt_correct = 1
                    rank3_g += 1
                    break

        for i,g in enumerate(gt):
            if i == 0 and g.lower() in filtered.lower():
                rank1 += 1 
                rank3 += 1
                pre_correct = 1
                break
            else:
                if g.lower() in filtered.lower():
                    rank3 += 1
                    pre_correct = 1
                    break

        if gpt_correct == 1 and pre_correct == 1:
            RR += 1
        elif gpt_correct == 1 and pre_correct == 0:
            WR += 1
        elif gpt_correct == 0 and pre_correct == 1:
            RW += 1
        else:
            WW += 1
    R3_g = rank3_g/len(gptresponse)
    R1 = rank1/len(gptresponse)
    R3 = rank3/len(gptresponse)
    consistent = response_con/len(gptresponse)
    RR = RR/len(gptresponse)
    RW = RW/len(gptresponse)
    WR = WR/len(gptresponse)
    WW = WW/len(gptresponse)
    print('R3_GPT:',R3_g)
    print('R1:',R1)
    print('R3:',R3)
    print('RR:',RR)
    print('RW:',RW)
    print('WR:',WR)
    print('WW:',WW)
    print('Consistent:',consistent)
