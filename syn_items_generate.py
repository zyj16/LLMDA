import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from util_ml_without_shuffle import get_prompt_conclass, parse_prompt2df, parse_result, get_unique_features, make_final_prompt
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

print(f"current file: {os.getcwd()}")
env_path = os.path.join(os.path.dirname(__file__), ".vscode", ".env")
load_dotenv(env_path)

data = pd.read_csv('./dataset/ml/step1/befor_syn/movies_sample.csv', index_col='index')
print(data.head())

CATEGORICAL_FEATURES = ['user_id', 'Action','Adventure','Animation',
                        'Children','Comedy','Crime','Documentary','Drama','Fantasy',
                        'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
NAME_COLS = ','.join(data.columns) + '\n'  
 
unique_categorical_features=get_unique_features(data, CATEGORICAL_FEATURES)
# unique_categorical_features['user_id']=[0,1,2]
unique_categorical_features['Action'] =[0,1] 
unique_categorical_features['Adventure'] =[0,1] 
unique_categorical_features['Animation'] =[0,1] 
unique_categorical_features['Children'] =[0,1]
unique_categorical_features['Comedy'] =[0,1]  
unique_categorical_features['Crime'] =[0,1]  
unique_categorical_features['Documentary'] =[0,1] 
unique_categorical_features['Drama'] =[0,1] 
unique_categorical_features['Fantasy'] =[0,1] 
unique_categorical_features['Film-Noir'] =[0,1] 
unique_categorical_features['Horror'] =[0,1]
unique_categorical_features['Musical'] =[0,1]
unique_categorical_features['Mystery'] =[0,1]
unique_categorical_features['Romance'] =[0,1]
unique_categorical_features['Sci-Fi'] =[0,1]
unique_categorical_features['Thriller'] =[0,1]
unique_categorical_features['War'] =[0,1]
unique_categorical_features['Western'] =[0,1]

N_CLASS = 942 # num_users
N_SAMPLES_PER_CLASS = 15  #
N_SET= 2
N_BATCH = 4
N_SAMPLES_TOTAL = N_SAMPLES_PER_CLASS*N_SET*N_BATCH # 
N_TARGET_SAMPLES=2


initial_prompt="""You are a data generator, please generate some data according to the given data format with the following requirements：
1. Generate samples according to the names of different columns given at the end of the prompt;
2. Apart from generating subsequent data according to the given data format, please do not generate any other text;
3. You can generate movie_tiitles that do not exist in the original input;
4. Do not generate new movie_titles in consecutive lines;
4. Do not copy existing data;
5. Generate identical items for multiple users as reasonably as possible to reduce data sparsity;
6. Do not periodically generate a movie_title repeatedly;
7. Generate at least one new piece of data for each user;
8. The generated data should not be the same as the input data；
9. Don't generate symbols in the movie_title;
10. Don't generate new user_ids, generated user_id must already exist in the original data;
11. If the movie_title contains commas, it needs to be enclosed in quotation marks;

Given data representing user interaction with different movies. The meaning of different column representations of data is:
user_id: the user in the recommendation who has several interactions with different items, there are 942 users in the dataset;
movie_title: the title of the movie, there is no symbol in the name;
Action: whether the type of the movie is Action, 0 represents False, 1 represents True;
Adventure: whether the type of the movie is Adventure, 0 represents False, 1 represents True;
Animation: whether the type of the movie is Animation, 0 represents False, 1 represents True;
Children: whether the type of the movie is Children, 0 represents False, 1 represents True;
Comedy: whether the type of the movie is Comedy, 0 represents False, 1 represents True;
Crime: whether the type of the movie is Crime, 0 represents False, 1 represents True;
Documentary: whether the type of the movie is Documentary, 0 represents False, 1 represents True;
Drama: whether the type of the movie is Drama, 0 represents False, 1 represents True;
Fantasy: whether the type of the movie is Fantasy, 0 represents False, 1 represents True;
Film-Noir: whether the type of the movie is Film-Noir, 0 represents False, 1 represents True;
Horror: whether the type of the movie is Horror, 0 represents False, 1 represents True;
Musical: whether the type of the movie is Musical, 0 represents False, 1 represents True;
Mystery: whether the type of the movie is Mystery, 0 represents False, 1 represents True;
Romance: whether the type of the movie is Romance, 0 represents False, 1 represents True;
Sci-Fi: whether the type of the movie is Sci-Fi, 0 represents False, 1 represents True;
Thriller: whether the type of the movie is Thriller, 0 represents False, 1 represents True;
War: whether the type of the movie is War, 0 represents False, 1 represents True;
Western: whether the type of the movie is Western, 0 represents False, 1 represents True.\n\n
"""
# user_movie_types: the genres of the movies that the user has interacted, it can not be changed during the data generation. You can generate the data for each user according to this column. 
# 11. When generating data, the last column of data remains unchanged for each user,don't generate data for the last column.
# 11. Users in each group have similar behavior patterns, and when generating data for a user in a group, it can be based on the behavior data of other users in the group.

# 11. For each row of data, the last column represents the movie types that the user has interacted with, and at least 50% of the newly generated movie types for each user must be the movie types that appear in the last column;
# 12. When generating data, the last column of data remains unchanged for each user,don't generate data for the last column.

numbering = [str(i) for i in range(1,944)] # 
prompt=get_prompt_conclass(initial_prompt, numbering, N_SAMPLES_PER_CLASS,N_CLASS,N_SET, NAME_COLS)

template1 = prompt
template1_prompt = PromptTemplate.from_template(template1)

# final_prompt, inputs_batchs = make_final_prompt(unique_categorical_features, "user_id", data, template1_prompt,
#                            N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS)


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
    | output_parser
)


def chunker(iterable, size):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, size)):
        yield chunk

def make_request_with_retry(chunk, max_retries=5, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = llm1.batch(chunk)
            return response
        except Exception as e:

            print(f"time out,try {attempt + 1} times retry...")
            time.sleep(retry_delay)  
    return None

while len(inter_text) < 300:
    print('input')
    final_prompt, inputs_batchs = make_final_prompt(
            unique_categorical_features, "user_id", data, template1_prompt,
           N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS
        )
    
    for i, chunk in enumerate(chunker(inputs_batchs, 2)):
        result = make_request_with_retry(chunk)
        if result is None:
            continue
        inter_text.extend(result)
     
        if i % 2 == 0: 
            time.sleep(5)

with open("./dataset/ml/step1/after_syn/withoutshuffle/inter_text_32_noitemid_300_step1.pkl", "wb") as file:
    pickle.dump(inter_text,file)
    
    


 
    
   


