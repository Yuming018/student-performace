from cmath import isnan
import csv
import pandas as pd
import numpy as np
import re
import argparse

def process(name):
    path = f'./{name}/'
    df = pd.read_csv(path + "original1.csv") 
    df.drop(columns=['Unnamed: 0'], inplace=True)
    # print(df)
    for i in range(len(df['predict'])):
        if pd.isna(df['predict'][i]):
            df['predict'][i] = ""

    df = df.to_numpy()
    df = [x for x in df.tolist()]
    
    data = ['\ufeff處理的字串', 'label', 'predict']

    with open(path + 'task1.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(data)
        for i in range(len(df)):
            writer.writerow(df[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-n', type=str, default='kaiyuan')
    args = parser.parse_args()
    
    process(args.file_name)