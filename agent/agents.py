import os
import time
import argparse
import json
import jsonlines
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
import pandas as pd
import numpy as np
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.regular_function import format_hist_kg_descriptions, format_cand_kg_descriptions, format_rec_kg_descriptions
from utils.rw_process import write_jsonl, read_jsonl
from utils.openai_api_request import api_request

class AnalystAgent():
    def __init__(self, args,  mode='rec'):
        self.memory = []
        self.info_list = []
        self.messages = []
        self.args = args
        self.mode = mode
        self.load_prompt()

    def load_prompt(self):
        if self.mode == "rec":
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prompt_wKG import (analyst_system_prompt, analyst_user_prompt,
                                                        analyst_memory_user_prompt,
                                                        analyst_build_memory,
                                                        analyst_build_memory1)
            else:
                raise ValueError("Invalid mode: {}".format(self.args.data_dir))
            self.analyst_system_prompt = analyst_system_prompt
            self.analyst_user_prompt = analyst_user_prompt
            self.analyst_memory_user_prompt = analyst_memory_user_prompt
            self.analyst_build_memory = analyst_build_memory
            self.analyst_build_memory1 = analyst_build_memory1
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def get_memory_snippet(self, k=2):
        """
        Build a compact summary of the last k rounds from self.info_list.
        This is injected into the memory-aware user prompt as `rejection_log`.
        """
        if not self.info_list:
            return "No previous rounds."

        recent = self.info_list[-k:]
        lines = []
        for info in recent:
            if self.mode == "rec":
                line = (
                    f"In Round {info['epoch']}:\n"
                    f"- Preference Summary: {info['summary']}\n"
                    f"- RankedList: {info['rec_items']}\n"
                    f"- Recommender reason: {info['rec_reason']}\n"
                )
                if info.get("user_reason") is not None:
                    line += (f"- User Feedback on Recommendations: {info['user_reason']}\n"
                             f"- User Decision on Recommendations: {info['decision']}\n"
                             )
            lines.append(line.strip())
        return "\n\n".join(lines)

    def act(self, data,  epoch):
        if self.mode != "rec":
            raise ValueError("Invalid mode: {}".format(self.mode))
        hist_kg_descriptions = format_hist_kg_descriptions(
            data.get('hist_descriptions', []), 10 , '2hop'
        )
        if len(self.info_list) == 0:
            system_prompt = self.analyst_system_prompt
            user_prompt = self.analyst_user_prompt.format(seq_str=data['seq_str'],
                                                      hist_kg_descriptions=hist_kg_descriptions
                                                     )
        else:
            system_prompt = self.analyst_system_prompt
            rejection_log = self.get_memory_snippet(k=2)
            user_prompt = self.analyst_memory_user_prompt.format(seq_str=data['seq_str'],
                                                                 hist_kg_descriptions=hist_kg_descriptions,
                                                                 rejection_log=rejection_log,
                                                     )
        if not self.messages:
            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            self.messages.append({"role": "user", "content": user_prompt})
        response, usage = api_request(self.messages, "analyst", self.args)
        if response is not None:
            self.messages.append({"role": "assistant", "content": response})
        return response


    def build_memory(self, info):
        if info['user_reason'] is not None:
            return self.analyst_build_memory.format(info['epoch'], info['summary'], info['rec_items'],
                                                    info['rec_reason'], info['user_reason'], info['decision'])
        else:
            return self.analyst_build_memory1.format(info['epoch'], info['summary'], info['rec_items'],
                                                    info['rec_reason'],)

    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]


class RecAgent():
    def __init__(self, args,  mode='rec'):
        self.memory = []
        self.info_list = []
        self.messages = []
        self.args = args
        self.mode = mode
        self.load_prompt()

    def load_prompt(self):
        if self.mode == "rec":
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prompt_wKG import (rec_system_prompt, rec_user_prompt,
                                                        rec_memory_user_prompt,
                                                        rec_build_memory,)
            else:
                raise ValueError("Invalid mode: {}".format(self.args.data_dir))
            self.rec_system_prompt = rec_system_prompt
            self.rec_user_prompt = rec_user_prompt
            self.rec_memory_user_prompt = rec_memory_user_prompt
            self.rec_build_memory = rec_build_memory
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))


    def get_memory_snippet(self, k=2):
        """
        Build a compact summary of the last k rounds from self.info_list.
        This is injected into the memory-aware user prompt as `rejection_log`.
        """
        if not self.info_list:
            return "No previous rounds."

        recent = self.info_list[-k:]
        lines = []
        for info in recent:
            if self.mode == "rec":
                line = (
                    f"In Round {info['epoch']}:\n"
                    f"- Preference Summary: {info['summary']}\n"
                    f"- RankedList: {info['rec_items']}\n"
                    f"- Recommender reason: {info['rec_reason']}\n"
                )
                if info.get("user_reason") is not None:
                    line += (f"- User's feedback on recommendations': {info['user_reason']}\n"
                             f"- User's decision for recommendations: {info['decision']}\n")
            lines.append(line.strip())
        return "\n\n".join(lines)


    def act(self, data,  epoch, reason=None, item=None):
        if self.mode != "rec":
            raise ValueError("Invalid mode: {}".format(self.mode))
        cand_kg_descriptions = format_cand_kg_descriptions(
            data.get('cand_descriptions', []), 20, '2hop'
        )
        if len(self.info_list) == 0:
            system_prompt = self.rec_system_prompt
            user_prompt = self.rec_user_prompt.format(seq_str=data['seq_str'],
                                                      summary=data.get('summary',''),
                                                      len_cans=data['len_cans'],
                                                      cans_str=data['cans_str'],
                                                      cand_kg_descriptions=cand_kg_descriptions,
                                                     )
        else:
            system_prompt = self.rec_system_prompt
            rejection_log = self.get_memory_snippet(k=2)
            user_prompt = self.rec_memory_user_prompt.format(epoch = epoch,
                                                             seq_str=data['seq_str'],
                                                             summary=data.get('summary',''),
                                                             len_cans=data['len_cans'],
                                                             cans_str=data['cans_str'],
                                                             cand_kg_descriptions=cand_kg_descriptions,
                                                             rejection_log = rejection_log,
                                                             )
        if not self.messages:
            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            self.messages.append({"role": "user", "content": user_prompt})
        response, usage = api_request(self.messages, "rec", self.args)
        if response is not None:
            self.messages.append({"role": "assistant", "content": response})
        return response

    def build_memory(self, info):
        return self.rec_build_memory.format(info['epoch'], info['summary'], info['rec_items'],
                                                 info['rec_reason'], info.get('user_reason', 'N/A'), info.get('decision', 'N/A'),)
    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]


class UserModelAgent():
    def __init__(self, args, mode='rec'):
        self.memory = []
        self.info_list = []
        self.messages = []
        self.args = args
        self.mode = mode
        self.load_prompt()

    def load_prompt(self):
        if self.mode == "rec":
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prompt_wKG import (user_system_prompt, user_user_prompt,
                                                        user_memory_user_prompt,
                                                        user_build_memory, user_build_memory_2)
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2

    def get_memory_snippet(self, k=2):
        """
        Build a compact summary of the last k rounds from self.info_list,
        from the user's perspective.
        """
        if not self.info_list:
            return "No previous rounds."

        recent = self.info_list[-k:]
        lines = []
        for info in recent:
            if self.mode == "rec":
                line = (
                    f"In Round {info['epoch']}:\n"
                    f"- Preference Summary: {info['summary']}\n"
                    f"- Ranked List: {info['rec_items']}\n"
                    f"- System reason: {info['rec_reason']}\n"
                )
                if info.get("user_reason") is not None:
                    line += (f"- Your feedback for the recommendations: {info['user_reason']}\n"
                             f"- Your preference for the recommendations: {info['decision']}\n"
                             )
            lines.append(line.strip())
        return "\n\n".join(lines)


    def act(self, data, epoch, reason=None, item=None):
        if self.mode != "rec":
            raise ValueError("act() only valid in mode='rec'")
        recommended_product_ids = []
        if item and len(item) >= 3:
            cand_descriptions = data.get('cand_descriptions', [])
            name_to_id = {desc['artist_name']: desc['artist_id']
                         for desc in cand_descriptions}

            for book_name in item:
                if book_name in name_to_id:
                    recommended_product_ids.append(name_to_id[book_name])

        rec_kg_descriptions = format_rec_kg_descriptions(
            data.get('cand_descriptions', []),
            recommended_product_ids, 'you', '2hop'
        )

        if len(self.info_list) == 0:
            system_prompt = self.user_system_prompt
            user_prompt = self.user_user_prompt.format(seq_str=data['seq_str'],
                                                       summary=data.get('summary',''),
                                                       rec_item=item,
                                                       rec_reason=reason,
                                                       rec_kg_descriptions=rec_kg_descriptions,
                                                       )
        else:
            system_prompt = self.user_system_prompt
            rejection_log = self.get_memory_snippet(k=2)
            user_prompt = self.user_memory_user_prompt.format(epoch = epoch,
                                                              seq_str=data['seq_str'],
                                                              summary=data.get('summary',''),
                                                              rejection_log=rejection_log,
                                                              rec_item=item,
                                                              rec_reason=reason,
                                                              rec_kg_descriptions=rec_kg_descriptions
                                                              )
        if not self.messages:
            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            self.messages.append({"role": "user", "content": user_prompt})
        response, usage = api_request(self.messages, "user", self.args)

        if response is not None:
            self.messages.append({"role": "assistant", "content": response})
        return response

    def build_memory(self, info):

        if info['user_reason'] is not None:
            return self.user_build_memory.format(info['epoch'], info['summary'],
                                                 info['rec_items'], info['rec_reason'],
                                                 info['user_reason'], info['decision'])
        else:
            return self.user_build_memory_2.format(info['epoch'], info['summary'], info['rec_items'], info['rec_reason'])

    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]
