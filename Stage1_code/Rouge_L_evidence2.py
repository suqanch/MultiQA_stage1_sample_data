import os, re
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
import tiktoken
from pathlib import Path

from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import nltk, re
# nltk.download('punkt')
import json
from rapidfuzz import fuzz
from tex_analysis import parse_latex_sections # input is tex file path
import matplotlib.pyplot as plt

path_Qwen = 'latex_paper_Qwen-max'
path_Gpt4 = 'latex_paper_gpt4.1'
path_Gpt5 = 'latex_paper_gpt5'
# path_Gpt4o = 'latex_paper_gpt4o'
# path_Gpt3_5 = 'latex_paper_gpt3.5_turbo'

# folders = [path_Qwen, path_Gpt4, path_Gpt5]
documents = ['Chulo', 'Docopilot', 'MdocSum']

path_Qwen_respond_lst = [os.path.join(path_Qwen, doc, 'processed_response.json') for doc in documents]
path_Gpt4_respond_lst = [os.path.join(path_Gpt4, doc, 'processed_response.json') for doc in documents]
path_Gpt5_respond_lst = [os.path.join(path_Gpt5, doc, 'processed_response.json') for doc in documents]

path_Qwen_tex_lst = [os.path.join(path_Qwen, doc, 'merged.tex') for doc in documents]
path_Gpt4_tex_lst = [os.path.join(path_Gpt4, doc, 'merged.tex') for doc in documents]
path_Gpt5_tex_lst = [os.path.join(path_Gpt5, doc, 'merged.tex') for doc in documents]

# path_original_tex = [os.path.join(path_Qwen, doc, 'merged.tex') for doc in documents]

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

ENC = tiktoken.encoding_for_model("gpt-4.1")
def token_len(text):
    return len(ENC.encode(text))

# def word_len_en(s):
#     return len(nltk.word_tokenize(s))

def clean_txt(text):
    """ sbu (\") to (") """
    return text.replace(r'\"', '"')


def fuzzy_match(section_name, document, threshold=80):
    """
    input: section_name (str), document (dict), threshold (int)

    Fuzzy match a section name against document sections.
    """
    result_lst = []
    # print(f"Fuzzy matching section: {section_name}")
    for doc_section in document.keys():
        fuzzy_score = fuzz.partial_ratio(section_name, doc_section)
        if fuzzy_score >= threshold:
            result_lst.append((doc_section, fuzzy_score))
    #get largest score section name
    if result_lst:
        # print(sorted(result_lst, key=lambda x: x[1], reverse=True))
        return max(result_lst, key=lambda x: x[1])[0]
    return None

_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def rougeL_recall(evidence, document, debug=False):

    """    
    document: {
    "Introduction": {
        "content": "...",
        "multimodal_elements_num": 0
    },
    }
    
    evidence:
    {
    "section": "§ M-DOCSUM-BENCH",
    "content": "...",
    "type": 'paragraph' 
    }

    Step1: 

    section_dict = {
    true_matched_section: score,
    }

    for title, value in document.items():
        content = value.get('content', '')
        score = score(evidence content vs content)
        score.append(float(_SCORER.score(target = content, prediction = gold_evidence)['rougeL'].recall))
    return score, section_name, true section name, gold evidence

    """
    # the evidence section name
    section_name = evidence.get('section', '')
    # reg only keep a-zA-Z0-9
    section_name = re.sub(r'[^a-zA-Z0-9 ]', '', section_name).strip().lower()
    evidence_content = clean_txt(evidence.get('content', ''))
    evidence_content_lst = evidence_content.split('...')
    matched_section_dict = {}
    matched_section_evidence_dict = {}
    for title, value in document.items():
        gold_content = value.get('content', '')
        #remove \n \r\n
        gold_content = gold_content.replace("\n", "").replace("\r", "")
        score = []
        for evidence_content_piece in evidence_content_lst:
            sc = _SCORER.score(target=evidence_content_piece, prediction=gold_content)['rougeL'].recall
            length = len(evidence_content_piece)
            score.append((sc, length))
        # score = max len evidence score
        sorted_scores = sorted(score, key=lambda x: x[1], reverse=True)
        
        matched_section_dict[title] = sorted_scores[0][0]
        matched_section_evidence_dict[title] = gold_content

    # get max score and gold title name
    if matched_section_dict:
        max_section_name = max(matched_section_dict, key=matched_section_dict.get)
        max_score = matched_section_dict[max_section_name]
        max_evidence = matched_section_evidence_dict[max_section_name]
        if debug and (max_section_name != section_name or max_score < 0.95):
            print(f"Debug: {section_name} mismatched with {max_section_name} (score: {max_score})")
            print(f"Evidence content: {evidence_content[:100]}...")
        return max_score, section_name, max_section_name, max_evidence

    return (0.0, section_name, None, "")


def fuzzy_compare_anchor(section_name, anchor_list, threshold=85):
    """
    Fuzzy match a section name against a list of anchor sections.

    input: section_name (str), anchor_list (list), threshold (int)
    output: bool    
    """
    result_lst = []
    # print(f"Fuzzy matching section: {section_name}")
    for anchor in anchor_list:
        fuzzy_score = fuzz.partial_ratio(section_name, anchor)
        if fuzzy_score >= threshold:
            result_lst.append((anchor, fuzzy_score))
    #get largest score section name
    if result_lst:
        # print(sorted(result_lst, key=lambda x: x[1], reverse=True))
        return True
    return False

ANCHOR_SECTIONS = {
    "intro_background": ["abstract", "introduction", "related work", "background", "preliminaries", "literature review"],
    "experiments": ["experiment", "evaluation", "results", "analysis", "experiment & analysis"],
    "conclusion": ["conclusion", "discussion", "summary", "closing remarks", "limitations"]
}

def fuzzy_match_section(document):
    """
    section_dict:
    {
    xxx: Introduction or Background,
    xxx: Introduction or Background
    xxx: Proposed Method,
    xxx: Experimental Results,
    xxx: Conclusion,
    xxx: Others
    }

    step1: find background, experimental, and conclusion sections

    step2:

    - i <= intro_end : ignore
    - intro_end < i < exp_start : Proposed Method
    - exp_start < i < concl_start : ignore
    - i >= concl_start : ignore
    - others : Others
    """

    section_dict = {}
    titles = [re.sub(r'[^a-zA-Z0-9 ]', '', section_name).lower() for section_name in document.keys()]
    # stage1 match intro, experiment, conclusion

    # Collect anchor indices
    intro_idxs, exp_idxs, concl_idxs = [], [], []
    for i, tnorm in enumerate(titles):
        if fuzzy_compare_anchor(tnorm, ANCHOR_SECTIONS["intro_background"]):
            intro_idxs.append(i)
        if fuzzy_compare_anchor(tnorm, ANCHOR_SECTIONS["experiments"]):
            exp_idxs.append(i)
        if fuzzy_compare_anchor(tnorm, ANCHOR_SECTIONS["conclusion"]):
            concl_idxs.append(i)

    intro_end   = max(intro_idxs) if intro_idxs else -1
    concl_start = min(concl_idxs) if concl_idxs else -1  
    exp_start   = min(exp_idxs)   if exp_idxs   else concl_start 
    if concl_start == -1:
        raise ValueError("Could not find conclusion section")

    title_to_category = {}

    for i, title in enumerate(titles):

        # Rule 1: up to last intro anchor -> Intro/Background
        if i <= intro_end and i in intro_idxs:
            title_to_category[title] = "Introduction or Background"
            continue

        # Rule 2: between last intro and first experiments -> Proposed Method
        if i > intro_end and i < exp_start:
            title_to_category[title] = "Proposed Method"
            continue

        # Rule 3: experiments block (from first experiments to before conclusion)
        if i >= exp_start and i in exp_idxs:
            # If there is a conclusion later, cap the block before it; otherwise till end
            title_to_category[title] = "Experimental Results"
            continue

        # Rule 4: anything at/after first Conclusion -> Conclusion
        if i >= concl_start and i in concl_idxs:
            title_to_category[title] = "Conclusion"
            continue

        # Fallback
        title_to_category[title] = "Others"

    return title_to_category


def process_responses_tex(respond_path_lst, tex_path_lst):

    """
    input: multiple respond and original text from one source LLM, multiple documents

    results of `parse_latex_sections`: 
    {
    "Introduction": {
        "content": "...",
        "multimodal_elements_num": 0
    },
    }

    one question corresponds to multiple evidences
    task1: For each question, count number of distinct sections, count evidence length
        add field for each question: num_sections

    task2: For each evidence, calculate ROUGE-L score with the original text
        add field for each evidence: rougeL_score, evidence_length

    ---------------------------------------------

    for tex files fuzzy match sections to:
    
    1. Introduction or Background
    2. Proposed Method
    3. Experimental Results
    4. Conclusion
    5. Others

    add section_categories to each evidence
    """

    results = []
    for respond_path, tex_path in zip(respond_path_lst, tex_path_lst):
        responses = load_json(respond_path)
        original_text = parse_latex_sections(tex_path)

        title_to_category = fuzzy_match_section(original_text)
        # print(f"Section categories: {title_to_category}")
        # fuzzy generate section_categories

        for qa in responses:
            # print(qa)
            question = qa.get('question', '')
            intent = qa.get('intent', '')
            evidences = qa.get('evidence', [])

            # Task 1: Count number of distinct sections and evidence length
            num_sections = len(set(evidence.get('section', '') for evidence in evidences))
            evidence_content = [len(clean_txt(evidence.get('content', ''))) for evidence in evidences]
            avg_evidence_len = sum(evidence_content) / len(evidence_content) if evidence_content else 0

            new_dict = {
                "question": question,
                "intent": intent,
                "num_sections": num_sections,
                "avg_evidence_len": avg_evidence_len,
                "evidences": []
            }

            # Task 2: Calculate ROUGE-L score with the original text
            # print(f"Processing question: {evidences}")
            for evidence in evidences:
                evidence_score, section_name, true_matched_section, gold_evidence = rougeL_recall(evidence, original_text, debug=False)
                # print(evidence_score)
                evidence_len = len(evidence.get('content', ''))

                new_dict['evidences'].append({
                    "section_name": section_name,
                    "fuzzy_matched_section": true_matched_section,
                    "true_category": title_to_category.get(true_matched_section, "Others"),
                    "content": evidence.get('content', ''),
                    "gold_paragraph": gold_evidence,
                    "max_rougeL_score": evidence_score,
                    "evidence_length": evidence_len,
                })

            results.append(new_dict)

    #save path  = respond_path_lst.parent, call it qa_stats.json

    #path_Qwen_respond_lst: latex_paper_Qwen-max/xxx/xxx.json
    # i want to save in latex_paper_Qwen-max/qa_stats.json
    save_path = Path(respond_path_lst[0]).parent.parent / 'qa_stats.json'
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {save_path}")

    return results

def bar_chart(result_dict, name, category):
    """
    input:
    true_category_count = {} or
    intent_count = {}

    name is model name
    category is "true_category_count" or "intent_count"

    save result to /plots/{name}_{category}.png
    """
    plt.figure(figsize=(10, 5))
    plt.bar(result_dict.keys(), result_dict.values())
    plt.title(f"{name} - {category}")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"./plots/bar_{name}_{category}.png")
    plt.close()

def box_plot(result_lst, name, category):
    """
    input:
    result_lst: list of numbers; [1,2,3,4, 9....]
    name: model name
    category: "evidence_count", "distinct_section_count", "evidence_len"

    save result to /plots/{name}_{category}.png
    """
    plt.figure(figsize=(10, 5))
    plt.boxplot(result_lst)
    plt.title(f"{name} - {category}")
    plt.xlabel(f"{category}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"./plots/box_{name}_{category}.png")
    plt.close()

def print_statistics(results, name):
    """
    input results
    results = [{
    "question": ...,
    "intent": [],
    "num_sections": 3,
    "avg_evidence_len": 0,
    "evidences": [
        {
        "section_name": ...,
        "true_matched_section": ...,
        "true_category": ...,
        "content": ...,
        "gold_paragraph": ...,
        "max_rougeL_score": ...,
        "evidence_length": ...
        },
    ]
    },]

    for the whole results;
        1. true_category counting
        2. count misclassified sections
        3. count RougeL < 0.95 number
        4. intent counting
        5. overall avg_evidence_len
        6. max/min average evidence length
        7. max/min/avg number of evidences(not conduct now)
    """
    true_category_count = {
        "Introduction or Background": 0,
        "Proposed Method": 0,
        "Experimental Results": 0,
        "Conclusion": 0,
        "Others": 0
    }

    intent_count = {
        "Descriptive": 0,
        "Procedural": 0,
        "Causal": 0,
        "Verificative": 0,
        "Comparative": 0,
        "Evaluative": 0
    }

    # min max avg
    avg_evidence_len = []

    evidence_length = []
    question_num = len(results)

    distinct_sections = []
    evidence_num = []

    misclassified_count = 0
    low_rouge_count = 0

    for result in results:
        avg_evidence_len.append(result.get("avg_evidence_len", 0))
        intent_lst = result.get("intent", [])

        distinct_sec = result.get("num_sections", 0)
        distinct_sections.append(distinct_sec)

        evidence_count = 0

        for intent in intent_lst:
            intent_count[intent] = intent_count.get(intent, 0) + 1
        for evidence in result.get("evidences", []):
            # Count true categories
            true_category = evidence.get("true_category", "Others")
            true_category_count[true_category] = true_category_count.get(true_category, 0) + 1
            evidence_length.append(evidence.get("evidence_length", 0))
            evidence_count += 1
            # Count misclassified sections
            if evidence.get("true_matched_section") != evidence.get("section_name"):
                misclassified_count += 1

            # Count low ROUGE-L scores
            if evidence.get("max_rougeL_score", 1) < 0.95:
                low_rouge_count += 1

        evidence_num.append(evidence_count)

    # print question num, 
    print(f"Total questions: {question_num}")
    
    # print min/max/avg evidence num
    print(f"Min evidence num: {min(evidence_num) if evidence_num else 0}")
    print(f"Max evidence num: {max(evidence_num) if evidence_num else 0}")
    print(f"Avg evidence num: {sum(evidence_num) / len(evidence_num) if evidence_num else 0}")
    # box_plot(evidence_num, name, "evidence_count")

    # print min/max/avg distinct section
    print(f"Min distinct sections: {min(distinct_sections) if distinct_sections else 0}")
    print(f"Max distinct sections: {max(distinct_sections) if distinct_sections else 0}")
    print(f"Avg distinct sections: {sum(distinct_sections) / len(distinct_sections) if distinct_sections else 0}")
    # box_plot(distinct_sections, name, "distinct_section_count")

    # print max/min/avg evidence length
    overall_avg_evidence_len = sum(evidence_length) / len(evidence_length) if evidence_length else 0
    max_evidence_len = max(evidence_length) if evidence_length else 0
    min_evidence_len = min(evidence_length) if evidence_length else 0
    
    print(f"Min evidence length: {min_evidence_len}")
    print(f"Max evidence length: {max_evidence_len}")
    print(f"Avg evidence length: {overall_avg_evidence_len}")
    # box_plot(evidence_length, name, "evidence_len")

    #print misclassified_count and low_rouge_count
    print(f"Misclassified sections: {misclassified_count}")
    print(f"Low ROUGE-L scores: {low_rouge_count}")

    # plot true_category_count and intent
    bar_chart(true_category_count, name, "true_category_count")
    bar_chart(intent_count, name, "intent_count")


    return evidence_num, distinct_sections, evidence_length


def box_plot3(Qwen_lst, Gpt4_lst, Gpt5_lst, name):
    plt.figure(figsize=(12, 6))
    plt.boxplot([Qwen_lst, Gpt4_lst, Gpt5_lst], tick_labels=["Qwen", "Gpt4.1", "Gpt5"])
    plt.title(f"Box Plot of - {name}")
    plt.ylabel(f"{name} Count")
    # plt.grid()
    plt.savefig(f"./plots/box_plot3_{name}.png")
    plt.close()


if __name__ == "__main__":

    print("Qwen3 results:")
    Qwen_result = process_responses_tex(path_Qwen_respond_lst, path_Qwen_tex_lst)
    Qwen_evidence_num, Qwen_distinct_sections, Qwen_evidence_length = print_statistics(Qwen_result, "Qwen3")

    print("\nGpt4.1 results:")
    Gpt4_result = process_responses_tex(path_Gpt4_respond_lst, path_Gpt4_tex_lst)
    Gpt4_evidence_num, Gpt4_distinct_sections, Gpt4_evidence_length = print_statistics(Gpt4_result, "Gpt4")

    print("\nGpt5 results:")
    Gpt5_result = process_responses_tex(path_Gpt5_respond_lst, path_Gpt5_tex_lst)
    Gpt5_evidence_num, Gpt5_distinct_sections, Gpt5_evidence_length = print_statistics(Gpt5_result, "Gpt5")

    # box plot evidence
    box_plot3(Qwen_evidence_num, Gpt4_evidence_num, Gpt5_evidence_num, "Evidence Count")
    box_plot3(Qwen_distinct_sections, Gpt4_distinct_sections, Gpt5_distinct_sections, "Distinct Section Count")
    box_plot3(Qwen_evidence_length, Gpt4_evidence_length, Gpt5_evidence_length, "Evidence Length")