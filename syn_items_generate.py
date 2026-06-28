import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from util_new import get_prompt_conclass, parse_prompt2df, parse_result, get_unique_features, make_final_prompt
import string
import random
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from openai import OpenAI
from itertools import islice
import time
import pickle
import json

class StatsTracker:
    

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        self.batch_stats = []
        self.script_start_time = None
        self.script_end_time = None

    def start(self):
      
        self.script_start_time = time.time()
        print(f"\n{'='*60}")
      
        print(f"{'='*60}\n")

    def record_batch(self, batch_id, ai_messages, duration_sec):
      
        batch_prompt = 0
        batch_completion = 0
        for msg in ai_messages:
            usage = getattr(msg, 'usage_metadata', None)
            if usage:
                batch_prompt += usage.get('input_tokens', 0)
                batch_completion += usage.get('output_tokens', 0)
            else:
             
                resp_meta = getattr(msg, 'response_metadata', {})
                token_usage = resp_meta.get('token_usage', resp_meta.get('usage', {}))
                batch_prompt += token_usage.get('prompt_tokens', token_usage.get('input_tokens', 0))
                batch_completion += token_usage.get('completion_tokens', token_usage.get('output_tokens', 0))

        batch_total = batch_prompt + batch_completion
        self.total_prompt_tokens += batch_prompt
        self.total_completion_tokens += batch_completion
        self.total_tokens += batch_total
        self.api_call_count += len(ai_messages)

        record = {
            'batch_id': batch_id,
            'num_messages': len(ai_messages),
            'prompt_tokens': batch_prompt,
            'completion_tokens': batch_completion,
            'total_tokens': batch_total,
            'duration_sec': round(duration_sec, 2)
        }
        self.batch_stats.append(record)

    
    def finish(self):
       
        self.script_end_time = time.time()

    def get_total_runtime(self):
        if self.script_start_time and self.script_end_time:
            return self.script_end_time - self.script_start_time
        return None


    def to_dict(self):
     
        runtime = self.get_total_runtime()
        return {
            'total_runtime_sec': round(runtime, 2) if runtime else None,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.script_start_time)) if self.script_start_time else None,
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.script_end_time)) if self.script_end_time else None,
            'api_call_count': self.api_call_count,
            'batch_count': len(self.batch_stats),
            'token_usage': {
                'total_prompt_tokens': self.total_prompt_tokens,
                'total_completion_tokens': self.total_completion_tokens,
                'total_tokens': self.total_tokens,
                'avg_prompt_tokens_per_call': round(self.total_prompt_tokens / self.api_call_count, 1) if self.api_call_count > 0 else 0,
                'avg_completion_tokens_per_call': round(self.total_completion_tokens / self.api_call_count, 1) if self.api_call_count > 0 else 0,
                'avg_tokens_per_call': round(self.total_tokens / self.api_call_count, 1) if self.api_call_count > 0 else 0,
            },
            'batch_details': self.batch_stats
        }

    def save_to_json(self, filepath):
       
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
      

env_path = os.path.join(os.path.dirname(__file__), ".vscode", ".env")
load_dotenv(env_path)


stats = StatsTracker()
stats.start()

data = pd.read_csv('./data/toys/train.csv')
data = data.drop(columns=['label'])
print(data.head())

CATEGORICAL_FEATURES = ['user_id','item_id']
NAME_COLS = ','.join(data.columns) + '\n'  
 
unique_categorical_features=get_unique_features(data, CATEGORICAL_FEATURES)


N_CLASS = 19377 
N_SAMPLES_PER_CLASS = 15  #
N_SET= 2
N_BATCH = 4
N_SAMPLES_TOTAL = N_SAMPLES_PER_CLASS*N_SET*N_BATCH 
N_TARGET_SAMPLES=2


#toys and games
initial_prompt="""You are a data generator, please generate some data according to the given data format with the following requirements：
1. Generate at least one new piece of data for each user;
2. The generated data should not be the same as the input data；
3. Don't generate new user_ids, generated user_id must already exist in the original data;

The meaning of different column representations of data is:
user_id: the user in the recommendation who has several interactions with different items;
item_id: the item in the recommendation which has interaction with several users;\n\n
"""

numbering = [str(i) for i in range(1,19377)] 
prompt=get_prompt_conclass(initial_prompt, numbering, N_SAMPLES_PER_CLASS,N_CLASS,N_SET, NAME_COLS)

template1 = prompt
template1_prompt = PromptTemplate.from_template(template1)

final_prompt, inputs_batchs = make_final_prompt(unique_categorical_features, "user_id", data, template1_prompt,
                           N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS)

input_df_all=pd.DataFrame()
synthetic_df_all=pd.DataFrame()
text_results = []

columns1=data.columns
columns2=list(data.columns)

err=[]


#aliyun
api_key=os.getenv("DASHSCOPE_API_KEY")

chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    # other params...
)
# api_key=os.getenv("DEEPSEEK_API_KEY")
# chatLLM = ChatOpenAI(
#     model='deepseek-chat',
#     openai_api_key=api_key,
#     openai_api_base='https://api.deepseek.com',
#     max_tokens=1024
# )

output_parser = StrOutputParser()

llm1 = (
    template1_prompt
    | chatLLM
)


def chunker(iterable, size):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, size)):
        yield chunk
def make_request_with_retry(chunk, batch_id=None, max_retries=5, retry_delay=5):

    for attempt in range(max_retries):
        try:
            batch_start = time.time()
            ai_messages = llm1.batch(chunk)
            batch_duration = time.time() - batch_start

            text_list = [msg.content for msg in ai_messages]

            if batch_id is not None:
                stats.record_batch(batch_id, ai_messages, batch_duration)

            return text_list, ai_messages
        except Exception as e:
            time.sleep(retry_delay)
    return None, None

inter_text = []
batch_counter = 0 



while len(inter_text) < 1000:
    print('input')
    final_prompt, inputs_batchs = make_final_prompt(
            unique_categorical_features, "user_id", data, template1_prompt,
           N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS
        )

    for i, chunk in enumerate(chunker(inputs_batchs, 2)):
        batch_counter += 1
        text_list, ai_messages = make_request_with_retry(chunk, batch_id=batch_counter)
        if text_list is None:
            continue
        inter_text.extend(text_list)

        if i % 2 == 0: 
            time.sleep(5)

# generate
print('input')
final_prompt, inputs_batchs = make_final_prompt(
            unique_categorical_features, "user_id", data, template1_prompt,
           N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS
        )

for i, chunk in enumerate(chunker(inputs_batchs, 10)):
    batch_counter += 1

    text_list, ai_messages = make_request_with_retry(chunk, batch_id=batch_counter)
    if text_list is None:
      
        continue
    inter_text.extend(text_list)

    if i % 2 == 0:  
        time.sleep(5)


stats.finish()
stats.print_summary()


stats_json_path = "./data/amazon_toys_and_games_generate_batch1_deepseek_stats.json"
stats.save_to_json(stats_json_path)

with open("./data/amazon_toys_and_games_generate_batch1_qwen.pkl", "wb") as file:
    pickle.dump(inter_text,file)


# with open("./data/book-crossing/book-crossing-deepseek.pkl", "rb") as file:
#     inter_text = pickle.load(file)
# print(len(inter_text))
# text_results = []
# all_df=[]
# for i in range(len(inter_text)):
  
#     text_results.append(inter_text[i])
#     # input_df = parse_prompt2df(final_prompt[i].text, split=NAME_COLS, inital_prompt=initial_prompt, col_name=columns1)
#     result_df = parse_result(inter_text[i], NAME_COLS, columns2, CATEGORICAL_FEATURES, unique_categorical_features)        
#     # input_df_all = pd.concat([input_df_all, input_df], axis=0)
#     # synthetic_df_all = pd.concat([synthetic_df_all, result_df], axis=0)
#     all_df.append(result_df)
    
# final_df =  pd.concat(all_df, axis=0)
# final_df.to_csv('./data/book-crossing/book-crossing-deepseek.csv', index_label='synindex')






 
    
   


