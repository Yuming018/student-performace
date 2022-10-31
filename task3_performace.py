from cmath import isnan
import csv
from pickle import NONE
import pandas as pd
import numpy as np
import re

def label():
    original_df = pd.read_csv("./suffle_test.csv") 
    original_df.drop(columns=['timestamp'], inplace=True)
    original_df.drop(columns=['algo 名稱'], inplace=True)
    original_df.drop(columns=['msno (用戶編號)'], inplace=True)
    original_df.drop(columns=['input text'], inplace=True)
    original_df.drop(columns=['confidence threshold'], inplace=True)
    original_df.drop(columns=['encoded device id'], inplace=True)
    original_df.drop(columns=['log date\n'], inplace=True)

    predict_df = pd.read_csv("./Task1_suffle_test.csv") 
    predict_df.drop(columns=['處理的字串'], inplace=True)
    predict_df.drop(columns=['label'], inplace=True)
    
    for i in range(len(predict_df['predict'])):
        if pd.isna(predict_df['predict'][i]):
            predict_df['predict'][i] = ""

    original_df = original_df.to_numpy()
    predict_df = predict_df.to_numpy()
    original_column = [x for x in original_df.tolist()]
    predict_df = [x for x in predict_df.tolist()]

    for i in range(len(original_df)):
        metadata = []
        temp = []
        for substr in re.finditer('metadata', original_column[i][1]):
            next_dot = original_column[i][1].find('"',substr.end()+4)
            metadata.append(original_column[i][1][substr.end()+4:next_dot])
            
            start_ = original_column[i][1].find('start',next_dot)
            if start_ == -1:
                break
            next_dot = original_column[i][1].find(',',start_+7)
            temp.append(original_column[i][1][start_+7:next_dot])

            end_ = original_column[i][1].find('end',next_dot)
            next_dot = original_column[i][1].find(',',end_+5)
            temp.append(original_column[i][1][end_+5:next_dot])
        # print(temp)
        # print(metadata)
        try:
            temp_str_list = '(' + temp[0] + " " + temp[1] + " " + metadata[0] + ')'
            for j in range(2, len(temp), 2):
                temp_str = '(' + temp[j] + " " + temp[j+1] + " " + metadata[int(j/2)] + ')'
                if temp_str not in temp_str_list:
                    temp_str_list = temp_str_list + " " + temp_str
            # print(temp_str_list)
            original_column[i][1] = temp_str_list
        except:
            original_column[i][1] = "" 
        # original_column[i].append(predict_df[i][0])
    
    data = ['\ufeff處理的字串', 'label', 'predict\n']

    with open('performace3.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(data)
        for i in range(len(original_column)):
            writer.writerow(original_column[i])

if __name__ == '__main__':
    label()