# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:47:28 2021

@author: kenneth
"""

import tushare as ts
import efinance as ef
import pandas as pd
import os
import talib as tl
from sqlalchemy import create_engine
import numpy as np

# start='20211224'
# end='20211227'

root_dir = 'f:\\play-stock'
code_dir = root_dir + '\\code'
data_dir = root_dir + '\\data'
model_dir = root_dir + '\\model'
temp_dir = root_dir + '\\temp'

til_date = '20220624'

os.chdir(root_dir)
ts.set_token('7b571c7a6118274b004dbdddde8c48a8ed5a656803e441584c4a5053')
pro = ts.pro_api()
count = 0
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')

engine= create_engine('sqlite:///data/ef_to0624stock.db')

print('getting data...')
print('total: ',len(df),' stocks')
for stockcode in df['ts_code'].values:
    if count < 4000:
        count = count + 1
        continue
    stockcode=stockcode[0:6]
    try:
        stock=ef.stock.get_quote_history(stock_codes=stockcode, end=til_date)
        if stock is None:
            continue
        stock.rename(columns={'股票名称':'stockname','股票代码':'stockcode','日期':'trade_date','开盘':'open',\
                  '收盘':'close','最高':'high','最低':'low','成交量':'vol','成交额':'val','振幅':'amplitude',\
                      '涨跌幅':'change rate','涨跌额':'change','换手率':'turnover'}, inplace = True)
        d = stock['trade_date'].values
        for i in range(len(d)):
            d[i] = d[i][0:4] + d[i][5:7]+d[i][8:10]
        # stock=stock[::-1].reset_index(drop=True)
 
        process = "\r[No:%4s %9s] "%(count,stockcode)
        stock['vol'] = stock['vol'].astype(np.double)
        stock['closeMA90'] = tl.MA(stock['close'].values, timeperiod = 90)
        stock['closeMA30'] = tl.MA(stock['close'].values, timeperiod = 30)
        stock['closeMA15'] = tl.MA(stock['close'].values, timeperiod = 15)
        stock['volMA90'] = tl.MA(stock['vol'].values, timeperiod = 90)
        stock['volMA30'] = tl.MA(stock['vol'].values, timeperiod = 30)
        stock['volMA15'] = tl.MA(stock['vol'].values, timeperiod = 15)
        stock['turnoverMA90'] = tl.MA(stock['turnover'].values,timeperiod = 90)
        stock['turnoverMA30'] = tl.MA(stock['turnover'].values,timeperiod = 30)
        stock['turnoverMA15'] = tl.MA(stock['turnover'].values,timeperiod = 15)
        stock.dropna(axis = 0, how = 'any', inplace = True)
        stock=stock.reset_index(drop = True)
        print(process, end='',flush=True)  
        stock.to_sql('stockdata', if_exists = 'append', index = False, con = engine)
    
        count = count + 1
    except:
        print('error get data: ', stockcode)
        break
    # if count >= 4000:
    #     break
    continue