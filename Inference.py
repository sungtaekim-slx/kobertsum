import json
import numpy as np
import pandas as pd
import time
import re
import sys
import os
import argparse
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
# python make_submission.py result_1209_1236_step_7000.candidate
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", type=str, required=True)
    parser.add_argument("-candidate_file", type=str, required=True)
    parser.add_argument("-output_path", type=str, default="./output")
    parser.add_argument("-output_csv", type=str, default="output.csv")

    args = parser.parse_args()
    createFolder(args.output_path)
    # test set
    with open(args.input_file, 'r') as json_file:
        json_list = list(json_file)

    tests = []
    for json_str in json_list:
        line = json.loads(json_str)
        tests.append(line)
    test_df = pd.DataFrame(tests)

    # 추론결과
    with open(args.candidate_file, 'r') as file:
        lines = file.readlines()
    # print(lines)
    test_pred_list = []
    for line in lines:
        sum_sents_text, sum_sents_idxes = line.rsplit(r'[', maxsplit=1)
        sum_sents_text = sum_sents_text.replace('<q>', ' ')
        sum_sents_idx_list = [ int(str.strip(i)) for i in sum_sents_idxes[:-2].split(', ')]
        test_pred_list.append({'sum_sents_tokenized': sum_sents_text, 
                            'sum_sents_idxes': sum_sents_idx_list
                            })

    result_df = pd.merge(test_df, pd.DataFrame(test_pred_list), how="left", left_index=True, right_index=True)
    result_df['summary'] = result_df.apply(lambda row: ' '.join(list(np.array(row['article_original'])[row['sum_sents_idxes']])) , axis=1)

    submit_df = pd.DataFrame()
    submit_df["id"] = result_df["id"]
    submit_df["summary"] = result_df["summary"]
    submit_df.to_csv(os.path.join(args.output_path, args.output_csv), index=False, encoding="utf-8")
    # submit_df = pd.read_csv("sample_submission.csv")
    # submit_df.drop(['summary'], axis=1, inplace=True)

    # print(result_df['id'].dtypes)
    # print(submit_df.dtypes)
    
    # result_df['id'] = result_df['id'].astype(int)
    # submit_df['summary'] = result_df['summary']
    # print(result_df['id'].dtypes)
    # # submit_df  = pd.merge(submit_df, result_df.loc[:, ['id', 'summary']], how="left", left_on="id", right_on="id")
    # print(submit_df.isnull().sum())
    # print(submit_df)
    # ## 결과 통계치 보기
    # # word
    # abstractive_word_counts = submit_df['summary'].apply(lambda x:len(re.split('\s', x)))
    # print(abstractive_word_counts.describe())

    # # export
    # now = time.strftime('%y%m%d_%H%M')
    # submit_df.to_csv(os.path.join(args.output_path, args.output_csv), index=False, encoding="utf-8")