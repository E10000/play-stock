# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:40:32 2021

@author: kenneth
"""
from dateutil import parser
import tushare as ts
import efinance as ef
import talib as tl
import pandas as pd
import numpy as np
import os
import time

root_dir = 'f:git-pro\\play-stock'
code_dir = root_dir + '\\code'
data_dir = root_dir + '\\data'
model_dir = root_dir + '\\model'
temp_dir = root_dir + '\\temp'

#os.chdir('f:\\python\\stock')
ts.set_token('7b571c7a6118274b004dbdddde8c48a8ed5a656803e441584c4a5053')
pro = ts.pro_api()

df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
# df = df[df['market'] !='北交所']
maxprice = []
avgprice = []
totalstock = []
missed = []
df = df[4000:]

print('getting data...')
print('total: ',len(df),' stocks')
for code in df['ts_code'].values:
    code = code[0:6]
    stock=ef.stock.get_quote_history(stock_codes=code, end='20220527')
    if stock is None:
        maxprice.append(10)
        avgprice.append(10)
        totalstock.append(10000)
        missed.append(code)
        continue
    if len(stock)==0:
        maxprice.append(10)
        avgprice.append(10)
        totalstock.append(10000)
        missed.append(code)
        continue    
    count = len(maxprice)

    maxprice.append(stock['最高'].max())
    avgprice.append(stock['最高'].mean())
    series = ef.stock.get_base_info(code)
    close = stock.iloc[-1]['收盘']
    total = series['流通市值']
    capital = total/close
    totalstock.append(capital)
    process = "\r[No:%4s %9s %19s] "%(count,code,capital)
    print(process, end='',flush=True)





df['maxprice'] = maxprice
df['avgprice'] = avgprice
df['totalshare'] = totalstock
 
df.to_pickle(data_dir + '\\basic_ef_data5.pkl')
print(df['totalshare'].max(), df['totalshare'].min())
print(missed)