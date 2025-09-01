from ctypes import util
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
import tiktoken
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertConfig, BertModel
import torch
# import numpy as np
import nltk, re
# nltk.download('punkt')
import json
import os

from FlagEmbedding import FlagAutoModel
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
path_Qwen = 'latex_paper_Qwen'
path_Gpt4 = 'latex_paper_gpt4.1'
path_Gpt5 = 'latex_paper_gpt5'

doc_lst = ['Chulo', 'Docopilot', 'MdocSum']
Qwen_doc_lst = [os.path.join(path_Qwen, doc, 'processed_response.json') for doc in doc_lst]
Gpt4_doc_lst = [os.path.join(path_Gpt4, doc, 'processed_response.json') for doc in doc_lst]
Gpt5_doc_lst = [os.path.join(path_Gpt5, doc, 'processed_response.json') for doc in doc_lst]
# path_original = os.path.join(path_Gpt4, 'Chulo/output.txt')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_txt(text):
    """ sbu (\") to (") """
    return text.replace(r'\"', '"')

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def draw_combine_boxplot(Qwen_score, GPT_score, Gpt5_score, save_path1):
    plt.figure(figsize=(10, 6))
    # separate the data into two groups
    data = [Qwen_score, GPT_score, Gpt5_score]
    plt.boxplot(data, labels=['Qwen-max-latest', 'Gpt4.1', 'Gpt5'])
    plt.title('Boxplot of Qwen-max-latest, Gpt4.1 and Gpt5 Scores')
    plt.ylabel('Scores')
    # save the figure
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved to {save_path1}")

def calculate_cosine_combined(questions, orginal_txt,  model_Q):

    """
    Calculate the cosine similarity for the given question and original text.
    return: (score, section_num, avg)
    """
    result_lst = []

    #lst item
    # {
    #     "question": question,
    #     "evidence": evidence,
    #     "score": score
    # }

    for qa in questions:
        question = qa.get("question", "")
        evidence_lst = [evidence.get("content", "") for evidence in qa.get("evidence", [])]
        # concat
        evidence = " ".join(evidence_lst)
        # for Qwen model
        with torch.no_grad():
            question_emb = model_Q.encode([question], prompt_name="query")
            evidence_emb = model_Q.encode([evidence], prompt_name="document")

        # print(f'shape: {question_emb.shape}, {evidence_emb.shape}')

        scores = model_Q.similarity(question_emb, evidence_emb)
        scores = scores.squeeze()
        print(scores)
        # if  tensor(0.7398)
        if scores.dim() == 0:        # scalar tensor
            scores = scores.unsqueeze(0)   # make it shape (1,)
        else:
            scores = scores.squeeze()      # normal squeeze for (N,)
        
        result_lst.append({
            "question": question,
            "evidence": evidence,
            "score": float(scores.item())
        })

    # avg scores
    return result_lst

def calculate_cosine_individual(questions, orginal_txt,  model_Q):

    """
    Calculate the cosine similarity for the given question and original text.
    return: (score, section_num, avg)
    """
    result_lst = []

    #lst item
    # {
    #     "question": question,
    #     "evidence": evidence,
    #     "score": score
    # }

    for qa in questions:
        question = qa.get("question", "")
        evidence_lst = [evidence.get("content", "") for evidence in qa.get("evidence", [])]

        # for Qwen model
        with torch.no_grad():
            question_emb = model_Q.encode([question], prompt_name="query")
            evidence_emb = model_Q.encode(evidence_lst)

        # print(f'shape: {question_emb.shape}, {evidence_emb.shape}')

        scores = model_Q.similarity(question_emb, evidence_emb)
        scores = scores.squeeze()
        print(scores)
        # if  tensor(0.7398)
        if scores.dim() == 0:        # scalar tensor
            scores = scores.unsqueeze(0)   # make it shape (1,)
        else:
            scores = scores.squeeze()      # normal squeeze for (N,)
        for q, evid, score in zip([question] * len(evidence_lst), evidence_lst, scores):
            result_lst.append({
                "question": q,
                "evidence": evid,
                "score": float(score.item())
            })

    # avg scores
    return result_lst


if __name__ == "__main__":

    # MODEL = "BAAI/bge-base-en-v1.5"   
    # Q_PREFIX = "Represent this sentence for searching relevant passages: "
    
    # print("Available prompts:", model_Q.prompts.keys())
    # Available prompts: dict_keys(['query', 'document'])
    # show prompt for query and document
    # print(model_Q.prompts["query"])
    # print(model_Q.prompts["document"])

    # model = FlagAutoModel.from_finetuned(
    #     MODEL,
    #     query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    #     use_fp16=True,
    #     devices=['cpu'],   # GPU ['cuda:0']
    # )
   
    model_Q = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')

    original_txt = None

    data_Q = []
    for path in Qwen_doc_lst:
        data_Q.extend(load_json(path))
    
    data_G = []
    for path in Gpt4_doc_lst:
        data_G.extend(load_json(path))

    data_G5 = []
    for path in Gpt5_doc_lst:
        data_G5.extend(load_json(path))

    print(f'Qwen3: {len(data_Q)}, Gpt4: {len(data_G)}, Gpt5: {len(data_G5)}')

    result_Q = calculate_cosine_individual(data_Q, original_txt, model_Q)
    print(len(result_Q))
    save_json("all_Qwen3_score2.json", result_Q)

    result_G = calculate_cosine_individual(data_G, original_txt, model_Q)
    print(len(result_G))
    save_json("all_Gpt4_score2.json", result_G)

    result_G5 = calculate_cosine_individual(data_G5, original_txt, model_Q)
    print(len(result_G5))
    save_json("all_Gpt5_score2.json", result_G5)

    # plot resultQ&G
    scoreQ = [item['score'] for item in result_Q]
    scoreG = [item['score'] for item in result_G]
    scoreG5 = [item['score'] for item in result_G5]
    print(f'Qwen3 score: max {max(scoreQ)}, min {min(scoreQ)}, avg {sum(scoreQ)/len(scoreQ)}')
    print(f'Gpt4 score: max {max(scoreG)}, min {min(scoreG)}, avg {sum(scoreG)/len(scoreG)}')
    print(f'Gpt5 score: max {max(scoreG5)}, min {min(scoreG5)}, avg {sum(scoreG5)/len(scoreG5)}')
    draw_combine_boxplot(scoreQ, scoreG, scoreG5, save_path1="Qwen3_Gpt4.1_boxplot2.png")

    # filter out top 5 scores
    top_5_Q = sorted(result_Q, key=lambda x: x['score'], reverse=True)[:5]
    top_5_G = sorted(result_G, key=lambda x: x['score'], reverse=True)[:5]
    top_5_G5 = sorted(result_G5, key=lambda x: x['score'], reverse=True)[:5]
    # filter out low 5 scores
    low_5_Q = sorted(result_Q, key=lambda x: x['score'])[:5]
    low_5_G = sorted(result_G, key=lambda x: x['score'])[:5]
    low_5_G5 = sorted(result_G5, key=lambda x: x['score'])[:5]

    # save to json
    save_json("top_5_Qwen3_2.json", top_5_Q)
    save_json("top_5_Gpt4_2.json", top_5_G)
    save_json("top_5_Gpt5_2.json", top_5_G5)
    save_json("low_5_Qwen3_2.json", low_5_Q)
    save_json("low_5_Gpt4_2.json", low_5_G)
    save_json("low_5_Gpt5_2.json", low_5_G5)

    #
