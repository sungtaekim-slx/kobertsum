import os
import sys
# os.chdir('./src')
sys.path.append("./src")

import re
#import MeCab
from bs4 import BeautifulSoup
import kss
import json
import numpy as np 
import pandas as pd
from tqdm import tqdm
import argparse
import pickle

import argparse
import time

from others.logging import init_logger
from prepro import data_builder

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

## 사용할 path 정의
# PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
# DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
# RAW_DATA_DIR = DATA_DIR + '/raw'
# JSON_DATA_DIR = DATA_DIR + '/json_data'
# BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = '../logs'
# LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 


# special_symbols_in_dict = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-']
# unused_tags = ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']
# def korean_tokenizer(text, unused_tags=None, print_tag=False): 
#     # assert if use_tags is None or unuse_tags is None
    
#     tokenizer = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ko-dic")
#     parsed = tokenizer.parse(text)
#     word_tag = [w for w in parsed.split("\n")]
#     result = []
    
#     if unused_tags:
#         for word_ in word_tag[:-2]:
#             word = word_.split("\t")
#             tag = word[1].split(",")[0]
#             if tag not in unused_tags:
#                 if print_tag:
#                     result.append((word[0], tag))
#                 else:
#                     result.append(word[0]) 
#     else:
#         for word_ in word_tag[:-2]:
#             word = word_.split("\t")
#             result.append(word[0]) 

#     return result

def number_split(sentence):
    # 1. 공백 이후 숫자로 시작하는 경우만(문자+숫자+문자, 문자+숫자 케이스는 제외), 해당 숫자와 그 뒤 문자를 분리
    num_str_pattern = re.compile(r'(\s\d+)([^\d\s])')
    sentence = re.sub(num_str_pattern, r'\1 \2', sentence)

    # 2. 공백으로 sentence를 분리 후 숫자인경우만 공백 넣어주기
    #numbers_reg = re.compile("\s\d{2,}\s")
    sentence_fixed = ''
    for token in sentence.split():
        if token.isnumeric():
            token = ' '.join(token)
        sentence_fixed+=' '+token
    return sentence_fixed

def noise_remove(text):
    text = text.lower()
    
    # url 대체
    # url_pattern = re.compile(r'https?://\S*|www\.\S*')
    # text = url_pattern.sub(r'URL', text)

    # html 삭제
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 숫자 중간에 공백 삽입하기
    # text = number_split(text)
    #number_pattern = re.compile('\w*\d\w*') 
#     number_pattern = re.compile('\d+') 
#     text = number_pattern.sub(r'[[NUMBER]]', text)
    

    # PUCTUACTION_TO_REMOVED = string.punctuation.translate(str.maketrans('', '', '\"\'#$%&\\@'))  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 중 적은것을 제외한 나머지를 삭제
    # text = text.translate(str.maketrans(PUCTUACTION_TO_REMOVED, ' '*len(PUCTUACTION_TO_REMOVED))) 

    # remove_redundant_white_spaces
    text = re.sub(' +', ' ', text)

    # tgt special token 으로 활용할 204; 314[ 315] 대체/삭제해줘서 없애주기
    text = re.sub('¶', ' ', text)
    text = re.sub('----------------', ' ', text)
    text = re.sub(';', '.', text)

    return text

def preprocessing(text, tokenizer=None):
    text = noise_remove(text)
    if tokenizer is not None:
        text = tokenizer(text)
        text = ' '.join(text)

    return text

def korean_sent_spliter(doc):
    sents_splited = kss.split_sentences(doc)
    if len(sents_splited) == 1:
        # .이나 ?가 있는데도 kss가 분리하지 않은 문장들을 혹시나해서 살펴보니
        # 대부분 쉼표나 가운데점 대신 .을 사용하거나 "" 사이 인용문구 안에 들어가있는 점들. -> 괜찮.
        # aa = sents_splited[0].split('. ')
        # if len(aa) > 1:
        #     print(sents_splited)
        return sents_splited
    else:  # kss로 분리가 된 경우(3문장 이상일 때도 고려)
        #print(sents_splited)
        for i in range(len(sents_splited) - 1):
            idx = 0
            # 두 문장 사이에 .이나 ?가 없는 경우: 그냥 붙여주기
            if sents_splited[idx][-1] not in ['.','?' ] and idx < len(sents_splited) - 1:
                sents_splited[idx] = sents_splited[idx] + ' ' + sents_splited[idx + 1] if doc[len(sents_splited[0])] == ' ' \
                                        else sents_splited[idx] + sents_splited[idx + 1] 
                del sents_splited[idx + 1]
                idx -= 1
        #print(sents_splited)
        return sents_splited


def create_json_files(df, data_type='train', target_summary_sent=None, path=''):
    NUM_DOCS_IN_ONE_FILE = 1000
    start_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))

    for start_idx in tqdm(start_idx_list):
        end_idx = start_idx + NUM_DOCS_IN_ONE_FILE
        if end_idx > len(df):
            end_idx = len(df)  # -1로 하니 안됨...

        #정렬을 위해 앞에 0 채워주기
        length = len(str(len(df)))
        start_idx_str = (length - len(str(start_idx)))*'0' + str(start_idx)
        end_idx_str = (length - len(str(end_idx-1)))*'0' + str(end_idx-1)

        file_name = os.path.join(f'{path}/{data_type}_{target_summary_sent}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json') if target_summary_sent is not None \
                    else os.path.join(f'{path}/{data_type}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json')
        
        json_list = []
        for i, row in df.iloc[start_idx:end_idx].iterrows():
            original_sents_list = [preprocessing(original_sent).split()  # , korean_tokenizer
                                    for original_sent in row['article_original']]

            summary_sents_list = []
            if target_summary_sent is not None:
                if target_summary_sent == 'ext':
                    summary_sents = row['extractive_sents']
                elif target_summary_sent == 'abs':
                    summary_sents = korean_sent_spliter(row['abstractive'])   
                summary_sents_list = [preprocessing(original_sent).split() # , korean_tokenizer
                                        for original_sent in summary_sents]

            json_list.append({'src': original_sents_list,
                              'tgt': summary_sents_list
            })
        #     print(json_list)
        #     break
        # break
        json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
        #print(json_string)
        with open(file_name, 'w') as json_file:
            json_file.write(json_string)

def do_format_to_model(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
def make_data(args):
    # args.data_path = "../"+args.data_path
    # args.log_path = "../"+args.log_path
    save_path = args.save_path
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    # import data
    with open(f'{args.data_path}/{args.train_data_name}', 'r') as json_file:
        train_json_list = list(json_file)
    with open(f'{args.data_path}/{args.test_data_name}', 'r') as json_file:
        test_json_list = list(json_file)

    trains = []
    for json_str in train_json_list:
        line = json.loads(json_str)
        trains.append(line)
    tests = []
    for json_str in test_json_list:
        line = json.loads(json_str)
        tests.append(line)

    # Convert raw data to df
    df = pd.DataFrame(trains)
    df['extractive_sents'] = df.apply(lambda row: list(np.array(row['article_original'])[row['extractive']]) , axis=1)

    # random split
    train_df = df.sample(frac=0.95,random_state=42) #random state is a seed value
    valid_df = df.drop(train_df.index)
    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)

    test_df = pd.DataFrame(tests)

    # save df
    train_df.to_pickle(f"{save_path}/train_df.pickle")
    valid_df.to_pickle(f"{save_path}/valid_df.pickle")
    test_df.to_pickle(f"{save_path}/test_df.pickle")
    print(f'train_df({len(train_df)}) is exported')
    print(f'valid_df({len(valid_df)}) is exported')
    print(f'test_df({len(test_df)}) is exported')
    
# python make_data.py -make bert -by abs
# Make bert input file for train and valid from df file
    
    
    for data_type in ['train', 'valid', 'test']:
        df = pd.read_pickle(f"{save_path}/{data_type}_df.pickle")

        ## make json file
        # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
        json_data_dir = f"{save_path}/{data_type}"
        if os.path.exists(json_data_dir):
            os.system(f"rm {json_data_dir}/*")
        else:
            os.mkdir(json_data_dir)

        
        
        ## Convert json to bert.pt files
        bert_data_dir = f"{save_path}/{data_type}"
        # createFolder(bert_data_dir)
        if os.path.exists(bert_data_dir):
            os.system(f"rm {bert_data_dir}/*")
        else:
            os.mkdir(bert_data_dir)

        create_json_files(df, data_type=data_type, path=save_path)
        
        args.dataset= data_type
        args.raw_path = json_data_dir
        
        args.save_path =  f"{save_path}/{data_type}"
        # createFolder(args.save_path)
        init_logger(args.log_path+"/"+args.log_file_name)
    # os.chdir('./src')    
        eval('data_builder.format_to_bert(args)')
        # os.system(f"python preprocess.py"
        #     + f" -mode format_to_bert -dataset {data_type}"
        #     + f" -raw_path {json_data_dir}"
        #     + f" -save_path {bert_data_dir}"
        #     + f" -log_file {args.log_path}/{args.log_file_name}"
        #     + f" -lower -n_cpus {args.n_cpus}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", default='monologg/koelectra-small-v3-discriminator', type=str)
    parser.add_argument("-target_summary_sent", default='ext', type=str)
    parser.add_argument("-n_cpus", default='1', type=int)
    parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-data_path", default='./data', type=str)
    parser.add_argument("-train_data_name", default='train.jsonl', type=str)
    parser.add_argument("-test_data_name", default='test.jsonl', type=str)
    parser.add_argument("-raw_path", default='./line_data', type=str)
    parser.add_argument("-save_path", default='./data/', type=str)
    parser.add_argument('-log_path', default='./logs', type=str)
    parser.add_argument('-log_file_name', default='preprocessing.log', type=str) 

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=1, type=int)    # 3
    parser.add_argument('-max_src_nsents', default=120, type=int)    # 100
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)    # 5
    parser.add_argument('-max_src_ntokens_per_sent', default=300, type=int)    # 200
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)    # 5
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)    # 500

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

   

    parser.add_argument('-dataset', default='')


    args = parser.parse_args()
    
    make_data(args)
    