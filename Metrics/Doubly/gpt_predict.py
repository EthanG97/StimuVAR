import json
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from openai import OpenAI



def gptapi(client,captions):
    input = [
        {"role": "system", "content": """Please assume the role of an expert in the emotional domain. We provide clues that may be related to the emotions of the viewer.\
         There are total 27 kinds of emotions, specifically, 0: 'Awkwardness', 1:'Empathic Pain', 2:'Fear', 3:'Anger', 4:'Sadness', 5:'Relief', 6:'Boredom', 7:'Joy', 8:'Aesthetic Appreciation', \
         9:'Adoration', 10:'Admiration', 11:'Amusement', 12:'Satisfaction', 13:'Disgust', 14:'Sexual Desire', 15:'Confusion', 16:'Romance', 17:'Craving', 18:'Horror', 19:'Excitement', 20:'Nostalgia', 21:'Awe (or Wonder)', \
         22:'Interest', 23:'Calmness', 24:'Surprise', 25:'Entrancement', 26:'Anxiety'. please return the answer in this way: \
         '''
         The viewer feels xxxx", because xxxxx
         '''
            """
            },
        {"role": "user", "content": \
         
            "After reading the below descriptions, how people might emotionally feel about the content and why. Only provide the one most likely emotion."+ captions[:300]
            
        
        }]
    
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages= input,
    temperature = 0, 
    max_tokens= 200)
    output = completion.choices[0].message.content
    return output

if __name__ == "__main__":
    client = OpenAI(api_key = '')
    parser = ArgumentParser()
    parser.add_argument('--response_path', type=str,default="",  help="path of the output file")


    args = parser.parse_args()
    files = [args.response_path.split('/')[-1]]
    outputfile = args.response_path.split('.jsonl')[0] + '_gpt.jsonl'

    file_path = args.response_path
    response = pd.read_json(path_or_buf=file_path,lines=True)
    if os.path.exists(outputfile):
        gptresponse = pd.read_json(path_or_buf=outputfile,lines=True)
        start = len(gptresponse)
    else:
        start = 0
    for i in tqdm(range(start,len(response))):
        video_id = response.iloc[i]['videoid']
        responses = response.iloc[i]['response']#[:-1]
        gt = response.iloc[i]['gt']
        if 'because' in responses:
            reasonss = []
            res = responses.split('.')
            for rep in res:
                if 'because' in rep:
                    import re
                    res= len(re.findall('(?=(because))', rep))
                    if res != 1:
                        reasonss.append(rep)
                    else:
                        filtered, reason = rep.split('because')
                        reasonss.append(reason)
            reason = '. '.join(reasonss)
        reason = reason.lower().replace(str(gt[0]).lower(), '')
        output = gptapi(client,reason)
        save = {'videoid':str(video_id), 'response': output, 'gt': gt}
        with open(outputfile, "a") as outfile:
            json_entry = json.dumps(save, ensure_ascii=False)
            outfile.write(json_entry + '\n')

