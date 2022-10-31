from cmath import isnan
import csv
import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score

def label():
    original_df = pd.read_csv("./suffle_test.csv") 
    original_df.drop(columns=['timestamp'], inplace=True)
    original_df.drop(columns=['algo 名稱'], inplace=True)
    original_df.drop(columns=['msno (用戶編號)'], inplace=True)
    original_df.drop(columns=['input text'], inplace=True)
    original_df.drop(columns=['confidence threshold'], inplace=True)
    original_df.drop(columns=['encoded device id'], inplace=True)
    original_df.drop(columns=['log date\n'], inplace=True)

    predict_df = pd.read_csv("./kai/task1.csv",encoding=' utf-8') 
    predict_df.drop(columns=['處理的字串'], inplace=True)
    predict_df.drop(columns=['label'], inplace=True)
    
    for i in range(len(predict_df['predict'])):
        if pd.isna(predict_df['predict'][i]):
            predict_df['predict'][i] = ""

    original_df = original_df.to_numpy()
    predict_df = predict_df.to_numpy()
    original_column = [x for x in original_df.tolist()]
    predict_df = [x for x in predict_df.tolist()]

    intention = []
    for i in range(len(original_df)):
        temp = []
        for substr in re.finditer('intention', original_column[i][1]):
            next_dot = original_column[i][1].find('"',substr.end()+3)
            if original_column[i][1][substr.end()+3:next_dot] not in temp:
                temp.append(original_column[i][1][substr.end()+3:next_dot])
            if original_column[i][1][substr.end()+3:next_dot] not in intention:
                intention.append(original_column[i][1][substr.end()+3:next_dot])
        try:
            temp_str = temp[0]
            for j in range(1,len(temp)):
                temp_str = temp_str + " " + temp[j]
            original_column[i][1] = temp_str
        except:
            original_column[i][1] = ""
        try:
            original_column[i].append(predict_df[i][0])
        except:
            original_column[i].append("")
    
    data = ['\ufeff處理的字串', 'label', 'predict']

    with open('performace.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(data)
        for i in range(len(original_column)):
            writer.writerow(original_column[i])

    return intention

def read_data(intention):
    df = pd.read_csv('./performace.csv')
    df.drop(columns=['處理的字串'], inplace=True)

    for i in range(len(df['label'])):
        if pd.isna(df['label'][i]):
            df['label'][i] = ""
        if pd.isna(df['predict'][i]):
            df['predict'][i] = ""
    df = df.to_numpy()
    
    orig ,pred = [], []
    for i in range(len(df)):
        temp ,temp2 = [] ,[]
        for j in range(len(intention)):
            if intention[j] not in df[i][0] or df[i][0] == "":
                temp.append(0)
            elif intention[j] in df[i][0]:
                temp.append(1)
            if intention[j] not in df[i][1] or df[i][1] == "":
                temp2.append(0)
            elif intention[j] in df[i][1]:
                temp2.append(1)
        orig.append(temp)
        pred.append(temp2) 
    return orig, pred

def score(orig, pred):
    macro_score = f1_score(orig, pred, average="macro")
    micro_score = f1_score(orig, pred, average="micro")
    print("macro_score : " + str(macro_score))
    print("micro_score : " + str(micro_score))

if __name__ == '__main__':
    intention = label()
    orig, pred = read_data(intention)
    score(orig, pred)