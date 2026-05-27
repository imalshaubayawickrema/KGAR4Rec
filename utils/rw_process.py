import json
import jsonlines
import pickle
import csv
import os


# Entities
USER = 'user'
ARTIST = 'artist'
TAG = 'tag'
GENRE = 'genre'
COUNTRY = 'country'
LANGUAGE = 'language'


# Relations
LISTEN = 'listen'
TAGGED = 'tagged'
FRIENDS_WITH = 'friends_with'
DESCRIBED_AS = 'described_as'
BELONG_TO = 'belong_to'
IS_FROM = 'is_from'
PERFORMS_IN = 'performs_in'



KG_RELATION = {
    USER: {
        LISTEN: ARTIST,
        TAGGED: TAG,
        FRIENDS_WITH: USER
    },
    ARTIST: {
        LISTEN: USER,
        DESCRIBED_AS: TAG,
        BELONG_TO: GENRE,
        IS_FROM: COUNTRY,
        PERFORMS_IN: LANGUAGE,
    },
    TAG: {
        TAGGED: USER,
        DESCRIBED_AS: ARTIST
    },
    GENRE: {
        BELONG_TO: ARTIST
    },
    COUNTRY: {
        IS_FROM: ARTIST,
    },
    LANGUAGE: {
        PERFORMS_IN: LANGUAGE,
    },
}

def get_entities():
    return list(KG_RELATION.keys())


def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]

def read_jsonl(file_path):
    data_list = []
    with open(file_path, "r",encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            data_list.append(json_data)
    return data_list

def write_jsonl(file_path, data_list):
    with jsonlines.open(file_path, 'w') as jsonl_file:
        for data in data_list:
            jsonl_file.write(data)
            
def append_jsonl(file_path, data):
    with jsonlines.open(file_path, 'a') as jsonl_file:
        jsonl_file.write(data)
            
def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data_list = json.load(json_file)
    return data_list
            
def write_json(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file,ensure_ascii=False, indent=4)  

def read_pk(file_path):
    data_list = []
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

def write_pk(file_path, data_list):
    with open(file_path, 'wb') as file:
        pickle.dump(data_list, file)


def read_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        data_reader = csv.DictReader(csvfile)
        data_list = [data for data in data_reader]
    return data_list

def write_csv(file_path, data_list):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data_list[0].keys())
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)


