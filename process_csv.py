from cmath import isnan
import csv
import pandas as pd
import numpy as np
import re

def process():
    path = './kai/'
    df = pd.read_csv(path + "original2.csv") 
    df.drop(columns=['Unnamed: 0'], inplace=True)
    # print(df)
    for i in range(len(df['predict'])):
        if pd.isna(df['predict'][i]):
            df['predict'][i] = ""

    df = df.to_numpy()
    df = [x for x in df.tolist()]
    
    data = ['\ufeff處理的字串', 'label', 'predict']

    with open(path + 'task2.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(data)
        for i in range(len(df)):
            writer.writerow(df[i])

if __name__ == '__main__':
    process()