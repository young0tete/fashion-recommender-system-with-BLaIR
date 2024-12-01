import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import os
import json
import sys
import heapq
from IPython.display import clear_output


# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check file existence
files = [
    ('user_review_with_LLM.csv', 'reviews'),
    ('item_new.csv', 'item'),
    ('user_review_cache.pth', 'user_review_cache'),
    ('user_LLM_cache.pth', 'user_LLM_cache'),
    ('item_cache.pth', 'item_cache'),
    ('sim_cache.pth', 'sim_cache'),
    ('ngcf.json', 'ngcf')
]

for file, var_name in files:
    if not os.path.exists(file):
        print(f"Error: The file '{file}' is missing.")
        sys.exit(1)  # No file -> error


# Load Datasets
reviews=pd.read_csv('user_review_with_LLM.csv')
item=pd.read_csv('item_new.csv')

user_review_cache = torch.load('user_review_cache.pth')
user_LLM_cache = torch.load('user_LLM_cache.pth')
item_cache = torch.load('item_cache.pth')
sim_cache = torch.load('sim_cache.pth')


# Load BLaIR: tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
model = AutoModel.from_pretrained("hyp1231/blair-roberta-base").to(device)

# text to embed function
def text_to_embed(text: str, tokenizer, model):
    # check NaN
    if pd.isna(text):
        print("NaN is included in the text datasets. Please check.")
        text = ""
        
    text=[text]
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)  # 데이터를 GPU로 이동
    
    with torch.no_grad():
        embeddings = model(**inputs, return_dict=True).last_hidden_state[:, 0]
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    return embeddings[0]


def calculate_cosine(mode: str, n: int, ngcf: dict, reviews: pd.DataFrame, item: pd.DataFrame) -> dict: #mode='review', 'LLM' #ngcf로 추출하는 item 개수
    #user_cache, item_cache, sim_cache -> comprehension
    if mode not in ("review", "LLM"):
        print("the mode should be \"review\" or \"LLM\".")
        return
    
    users_list=list(ngcf.keys())
    
    if mode=="review":
        user_cache=user_review_cache
    else:
        user_cache=user_LLM_cache
    
    #embedding and save user text
    for idx, user in enumerate(users_list):
        if user not in user_cache.keys():
            user_cache[user]=text_to_embed(reviews[reviews['user_id']==user][mode].iloc[0], tokenizer, model)
            
        for item_rank in range(n):
            #embedding and save item text
            item_id, score = ngcf[user][str(item_rank)]
            if item_id not in item_cache.keys():
                item_cache[item_id]=text_to_embed(item[item['item_id']==item_id]['item_text'].iloc[0], tokenizer, model)
            
            #calculate cosine similarity
            if (user, item_id) not in sim_cache.keys():
                sim_cache[(user, item_id)]=user_cache[user]@item_cache[item_id]
        
        clear_output(wait=True)
        print(f"Progress: {idx+1}/{len(users_list)}")

    
    #save cache
    torch.save(item_cache, 'item_cache.pth')
    torch.save(sim_cache, 'sim_cache.pth')
    if mode=="review":
        torch.save(user_cache, 'user_review_cache.pth')
    else:
        torch.save(user_cache, 'user_LLM_cache.pth')
        

            
    return sim_cache

def blending(n: int, k: int, beta: float, ngcf: dict, sim_cache: dict) -> dict:
    users_list=list(ngcf.keys())
    
    final_result={}

    for user in users_list:
        q=[] #priority queue
        predicted_items=[]
        for item_rank in range(n):
            item, ngcf_score = ngcf[user][str(item_rank)]
            nlp_score = sim_cache[(user, item)]
            final_score = ngcf_score*beta + nlp_score*(1-beta)

            # push to the heapq 
            heapq.heappush(q, (-final_score, item))

        # extract top-k items
        for i in range(k):
            minus_score, item=heapq.heappop(q)
            predicted_items.append(item)

        # save the result
        final_result[user]=predicted_items
        
    return final_result