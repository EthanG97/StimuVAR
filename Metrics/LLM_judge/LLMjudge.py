from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json
import argparse


def gptapi(client,captions):
    input = [
        {"role": "system", "content": """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.


"""
            },
        {"role": "user", "content": 
         
            """
Now here are the question and answer.

Question: how people might emotionally feel about the content and why. Only provide the one most likely emotion
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.""".format(answer = captions)
            
        
        }]
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages= input,
    temperature = 0, 
    max_tokens= 200)
    output = completion.choices[0].message.content
    return output



def main(filename1, outputfile):
    # Read the JSONL file
    response = pd.read_json(path_or_buf=filename1, lines=True)

    # Initialize OpenAI client
    client = OpenAI(api_key='sk-proj-u1OBgYYZB7QML8AnlP9OT3BlbkFJ1eypBKRNuHj13eVlsARM') # OpenAI key

    response_sample = response.sample(n=100, random_state=42)

    for index, row in tqdm(response_sample.iterrows(), total=response_sample.shape[0]):
        video_id = row['videoid']
        responses = row['response']
        
        # Get the output from gptapi
        output = gptapi(client, responses)
        
        # Prepare the save dictionary
        save = {'videoid': str(video_id), 'response': output}
        
        # Append to the output file

        with open(outputfile, "a", encoding='utf-8') as outfile:
            json_entry = json.dumps(save, ensure_ascii=False)
            outfile.write(json_entry + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some JSONL data and interact with OpenAI.')
    parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
    parser.add_argument('output_file', type=str, help='Path to the output JSONL file')

    args = parser.parse_args()
    main(args.input_file, args.output_file)


