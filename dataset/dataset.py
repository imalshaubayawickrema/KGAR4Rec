import torch
import os
import torch.utils.data as data
import gzip
import numpy as np
import pandas as pd
import random
from easydict import EasyDict as edict


class Dataset(data.Dataset):
    def __init__(self, data_dir=r'./data/lastfm', stage=None, cans_num=10, sep=", ", no_augment=True):
        self.data_dir = data_dir
        self.cans_num = cans_num
        self.stage = stage
        self.sep = sep
        self.aug = (stage == 'train') and not no_augment
        if 'lastfm' in self.data_dir:
            self.padding_item_id = 4606
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'], temp['next'])
        cans_name = [self.item_id2name[can] for can in candidates]
        sample = {
            'id': i,
            'userID': temp['userID'],
            'seq': temp['seq'],
            'seq_unpad': temp['seq_unpad'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'] - 1,
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': self.cans_num,
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        if 'liked_items' in temp:
            sample['liked_ids'] = temp['liked_ids']
            sample['liked_items'] = temp['liked_items']
            sample['liked_str'] = self.sep.join(temp['liked_items'])
        else:
            sample['liked_ids'] = []
            sample['liked_items'] = []
            sample['liked_str'] = ''

        if 'disliked_items' in temp:
            sample['disliked_ids'] = temp['disliked_ids']
            sample['disliked_items'] = temp['disliked_items']
            sample['disliked_str'] = self.sep.join(temp['disliked_items'])
        else:
            sample['disliked_ids'] = []
            sample['disliked_items'] = []
            sample['disliked_str'] = ''

        if 'seq_words' in temp:
            sample['seq_words'] = temp['seq_words']

        return sample

    def negative_sampling(self, seq_unpad, next_item):
        canset = [i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i != next_item]
        candidates = random.sample(canset, self.cans_num - 1) + [next_item]
        random.shuffle(candidates)
        return candidates

    def check_files(self):
        self.item_id2name = self.get_music_id2name()
        if self.stage == 'train':
            filename = "Train_data.df"
        elif self.stage == 'val':
            filename = "Valid_data.df"
        elif self.stage == 'test':
            filename = "Test_data.df"
        data_path = os.path.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)

    def _process_lastfm(self, df):

        liked_items_list = []
        liked_id_list = []
        disliked_items_list = []
        disliked_id_list = []

        for idx, row in df.iterrows():
            seq_unpad = row['seq_unpad']
            seq_title = row['seq_title']

            liked = []
            liked_id = []
            disliked = []
            disliked_id = []

            for item_id, item_name in zip(seq_unpad, seq_title):
                liked.append(item_name)
                liked_id.append(item_id)

            liked_items_list.append(liked)
            liked_id_list.append(liked_id)
            disliked_items_list.append(disliked)
            disliked_id_list.append(disliked_id)

        df['liked_ids'] = liked_id_list
        df['liked_items'] = liked_items_list
        df['disliked_ids'] = disliked_id_list
        df['disliked_items'] = disliked_items_list

        return df

    def get_music_id2name(self):
        music_id2name = dict()
        item_path = os.path.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r', encoding='utf-8-sig') as f:
            for lineno, l in enumerate(f, start=1):
                l = l.strip()
                if not l:
                    continue

                parts = l.split('::', 1)
                if len(parts) != 2:
                    print(f"[WARN] Skipping malformed line {lineno}: {l}")
                    continue

                try:
                    music_id2name[int(parts[0])] = parts[1].strip()
                except ValueError:
                    print(f"[WARN] Invalid ID at line {lineno}: {parts[0]}")
        return music_id2name

    def session_data4frame(self, datapath, music_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]

        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x

        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)

        def seq_to_title(x):
            return [music_id2name[x_i] for x_i in x]

        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)

        def next_item_title(x):
            return music_id2name[x]

        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        if 'lastfm' in self.data_dir:
            train_data = self._process_lastfm(train_data)
        return train_data
