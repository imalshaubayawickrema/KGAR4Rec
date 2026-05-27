import os
import time
import argparse
import json
import jsonlines
import pandas as pd
from collections import Counter

from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
import math

from dataset.dataset import Dataset

from utils.regular_function import split_user_response, split_rec_ranking, split_analyst_response
from utils.rw_process import *
from agent.agents import RecAgent, UserModelAgent, AnalystAgent

from dataset.lastfm_kg_data import build_vocabs
from dataset.lastfm_kg import KnowledgeGraphLastFM
from dataset.lastfm_kg_processor import KGPathProcessor

finish_num = 0
total = 0

sum_hit1 = 0
sum_hit5 = 0
sum_hit10 = 0
sum_ndcg5 = 0.0
sum_ndcg10 = 0.0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--url', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--max_retry_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=303)
    parser.add_argument('--mp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument("--save_info", action="store_true")
    parser.add_argument("--save_rec_dir", type=str, default=None)
    parser.add_argument("--save_user_dir", type=str, default=None)
    parser.add_argument("--save_analyst_dir", type=str, default=None)
    return parser.parse_args()

def recommend(data, args):
    start_time = time.time()
    analyst_agent = AnalystAgent(args, 'rec')
    rec_agent = RecAgent(args, 'rec')
    user_agent = UserModelAgent(args, 'rec')
    flag = False
    epoch = 1
    rec_items = None
    new_data_list = []
    while flag == False and epoch <= args.max_epoch:
        # analyst agent
        analyst_agent_response = analyst_agent.act(data, epoch)
        analyst_summary = split_analyst_response(analyst_agent_response)

        data['summary'] = analyst_summary
        # rec agent
        max_retries = 5
        attempt = 0

        while attempt < max_retries:
            rec_agent_response = rec_agent.act(data,epoch)
            rec_reason, rec_items = split_rec_ranking(rec_agent_response)
            if len(rec_items) < 10:
                rec_agent_response = rec_agent.act(data,epoch)
                rec_reason, rec_items = split_rec_ranking(rec_agent_response)

            if rec_items and len(rec_items) == 10:
                break
            attempt += 1
        else:
            # fallback: use available parsed items + fill from candidate list
            print("[Warning] RecAgent failed to return exactly 10 items after retries.")

            candidate_items = data["cans_name"] if "cans_name" in data else data["cans_str"].split(", ")
            correct_item = data["correct_answer"]

            rec_items = list(dict.fromkeys(rec_items))  # remove duplicates, preserve order

            remaining = [c.strip() for c in candidate_items
                         if c.strip() not in rec_items and c.strip() != correct_item]
            random.shuffle(remaining)
            for cand in remaining:
                rec_items.append(cand)
                if len(rec_items) == 10:
                    break
            rec_items = rec_items[:10]
        if args.max_epoch == 1:
            new_data = {'id': data['id'], 'seq_name': data['seq_name'], 'cans_name': data['cans_name'],
                     'correct_answer': data['correct_answer'], 'epoch': epoch, 'summary': data.get('summary', ''), 'rec_reason': rec_reason,
                    'ranked_rec': rec_items, 'user_reason': None, 'user_decision': None,
                    "hist_kg2":[f"{e['artist_name']}: {e['text_2hop']}" for e in data["hist_descriptions"]],
                    "cand_kg2":[f"{e['artist_name']}: {e['text_2hop']}" for e in data["cand_descriptions"]],
                    }
            new_data_list.append(new_data)
            memory_info = {"epoch": epoch, "summary":analyst_summary,"rec_items": rec_items, "rec_reason": rec_reason, "user_reason": None, "decision": None} #  "facet_json": data.get("facet_json","None")
            rec_agent.update_memory(memory_info)
            user_agent.update_memory(memory_info)
            analyst_agent.update_memory(memory_info)
            break

        # user agent
        while True:
            user_agent_response = user_agent.act(data, epoch, rec_reason, rec_items)
            evaluations, num_liked, flag = split_user_response(user_agent_response, rec_items) # facet
            if len(evaluations) < 10:
                print(
                    f"Warning: feedback provided for only {len(evaluations)}/10 items (user_id={data.get('id', '?')}, epoch={epoch})")
            if flag is not None:
                break

        # save
        new_data = {'id': data['id'], 'seq_name': data['seq_name'], 'cans_name': data['cans_name'],
                    'correct_answer': data['correct_answer'], 'epoch': epoch, 'summary': data.get('summary', ''),
                    'ranked_rec': rec_items, "user_reason": [f"{e['item']}: {e['feedback']}" for e in evaluations],
                    "user_decision": [f"{e['item']}: {e['decision']}" for e in evaluations],
                    "hist_kg2":[f"{e['artist_name']}: {e['text_2hop']}" for e in data["hist_descriptions"]],
                    "cand_kg2":[f"{e['artist_name']}: {e['text_2hop']}" for e in data["cand_descriptions"]],
                     }
        new_data_list.append(new_data)

        memory_info = {"epoch": epoch, "summary": analyst_summary,"rec_reason": rec_reason, "rec_items": rec_items,  "user_reason": [f"{e['item']}: {e['feedback']}" for e in evaluations],
                       "decision": [f"{e['item']}: {e['decision']}" for e in evaluations],}
        rec_agent.update_memory(memory_info)
        user_agent.update_memory(memory_info)
        analyst_agent.update_memory(memory_info)
        # update
        if flag:
            break

        epoch += 1
    end_time = time.time()
    print("recommendation time = ", end_time - start_time, flush=True)
    # save
    if args.save_info:
        analyst_file_path = os.path.join(args.save_analyst_dir, f"{data['id']}.jsonl")
        rec_file_path = os.path.join(args.save_rec_dir, f"{data['id']}.jsonl")
        user_file_path = os.path.join(args.save_user_dir, f"{data['id']}.jsonl")
        rec_agent.save_memory(rec_file_path)
        user_agent.save_memory(user_file_path)
        analyst_agent.save_memory(analyst_file_path)

    metrics = compute_metrics(rec_items, data['correct_answer'])
    return new_data_list, metrics, args


def compute_metrics(ranking, correct_item):
    """
    ranking: list[str] in order of predicted preference (best first)
    correct_item: str (ground truth next artist)

    Returns dict with hit@1, hit@5, hit@10, ndcg@5, ndcg@10
    """

    correct_norm = correct_item.lower().strip()
    pos = None
    for i, cand in enumerate(ranking):
        if cand.lower().strip() == correct_norm:
            pos = i + 1  # 1-based rank
            break

    def hit_at(k):
        return 1 if (pos is not None and pos <= k) else 0

    def ndcg_at(k):
        if pos is None or pos > k:
            return 0.0
        return 1.0 / math.log2(pos + 1)

    return {
        "hit1": hit_at(1),
        "hit5": hit_at(5),
        "hit10": hit_at(10),
        "ndcg5": ndcg_at(5),
        "ndcg10": ndcg_at(10),
    }

def setcallback(x):
    global finish_num, total
    global sum_hit1, sum_hit5, sum_hit10, sum_ndcg5, sum_ndcg10
    global token_rows

    data_list, metrics, args = x
    for data in data_list:
        output_file = os.path.join(args.output_dir, args.output_file)
        append_jsonl(output_file, data)
    finish_num += 1
    # accumulate
    sum_hit1 += metrics["hit1"]
    sum_hit5 += metrics["hit5"]
    sum_hit10 += metrics["hit10"]
    sum_ndcg5 += metrics["ndcg5"]
    sum_ndcg10 += metrics["ndcg10"]

    # running averages
    curr_hit1 = sum_hit1 / finish_num
    curr_hit5 = sum_hit5 / finish_num
    curr_hit10 = sum_hit10 / finish_num
    curr_ndcg5 = sum_ndcg5 / finish_num
    curr_ndcg10 = sum_ndcg10 / finish_num

    summary = {'total_users': total,
               'HR@1': curr_hit1,
               'HR@5': curr_hit5,
               'HR@10': curr_hit10,
               'NDCG@5': curr_ndcg5,
               'NDCG@10': curr_ndcg10,
               }

    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("==============")
    print(f"processed so far = {finish_num} / {total}")
    print(f"current avg hit@1   = {curr_hit1}")
    print(f"current avg hit@5   = {curr_hit5}")
    print(f"current avg hit@10  = {curr_hit10}")
    print(f"current avg ndcg@5  = {curr_ndcg5}")
    print(f"current avg ndcg@10 = {curr_ndcg10}")
    print("==============")


def main(args):
    if args.save_info and args.save_rec_dir is not None and not os.path.exists(args.save_rec_dir):
        os.makedirs(args.save_rec_dir)
    if args.save_info and args.save_user_dir is not None and not os.path.exists(args.save_user_dir):
        os.makedirs(args.save_user_dir)
    if args.save_info and args.save_analyst_dir is not None and not os.path.exists(args.save_analyst_dir):
        os.makedirs(args.save_analyst_dir)
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = Dataset(args.data_dir, args.stage, args.cans_num, args.sep, True)
    global total
    data_list = []
    for data in dataset:
        data_list.append(data)

    ds = build_vocabs(args.data_dir, dataset.session_data)
    ds_path = os.path.join(args.data_dir, "ds.pkl")
    with open(ds_path, "wb") as f:
        pickle.dump(ds, f)

    kg = KnowledgeGraphLastFM(ds)
    kg_path = os.path.join(args.data_dir, "kg.pkl")
    with open(kg_path, "wb") as f:
        pickle.dump(kg, f)
    kg.compute_degrees()
    processor = KGPathProcessor(kg, ds)

    for idx,d in enumerate(data_list):
        uid = idx
        analyst_result = processor.process_user_history(uid, d['liked_ids'], d['disliked_ids'])
        d['hist_descriptions'] = analyst_result['historical_descriptions']
        rec_agent_results = processor.process_candidate_items(uid, d['cans'], d['liked_ids'], d['disliked_ids'])
        d['cand_descriptions'] = rec_agent_results['candidate_descriptions']

    total = len(data_list)
    for data in data_list:
        setcallback(recommend(data, args))

if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)
