# -*- coding: utf-8 -*-
"""
Created on Thu May 12 20:00:01 2022

@author: kenneth
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import efinance as ef
import talib as tl
import tushare as ts
import datetime
import sqlite3

root_dir = 'f:\\git-pro\\play-stock'
code_dir = root_dir + '\\code'
data_dir = root_dir + '\\data'
model_dir = root_dir + '\\model'
temp_dir = root_dir + '\\temp'

start = '20220706'
end = '20220706'
ts.set_token('7b571c7a6118274b004dbdddde8c48a8ed5a656803e441584c4a5053')
pro = ts.pro_api()
batch_size = 256
market = ''
window = 10
ratio = []
codeList = []
nameList = []
dateList = []
closeList = []
marketList = []
changeList = []
nextHigh = []
nextLow = []
nextClose = []
nextOpen = []
futureHigh = []
trainX = []
showResult = True
extension = 5
inDBdate = '20220624'

contrue = sqlite3.connect(data_dir + '\\ef_to0624stock.db')

df = pd.read_pickle(data_dir + "\\basic_ef_data.pkl")
max_shares = df['totalshare'].max()

def daysBetween(d1, d2):
    return (parser.parse(d2) - parser.parse(d1)).days

def is_trade_date(dt):
    if dt in cal_dates['cal_date'].values:
        return cal_dates[cal_dates['cal_date']==dt].is_open.values[0]
    else:
        return False

def next_day(today):
    newdate=datetime.datetime.strptime(today,'%Y%m%d')+datetime.timedelta(days=1)
    return newdate.strftime('%Y%m%d')

def previous_day(today):
    newdate=datetime.datetime.strptime(today,'%Y%m%d')+datetime.timedelta(days=-1)
    return newdate.strftime('%Y%m%d')

def next_trade_day(today):
    day=next_day(today)
    while not is_trade_date(day):
        day = next_day(day)
    return day

def previous_trade_day(today):
    day=previous_day(today)
    while not is_trade_date(day):
        day = previous_day(day)
    return day

def get_true_stock(code, startdate, enddate):
    if enddate > inDBdate:
        stock=ef.stock.get_quote_history(stock_codes=code, beg=str(int(startdate)-10000), end=enddate)
        if stock is None:
            return None
        if len(stock) < 90 + window + extension:
            return None
        stock.rename(columns={'股票名称':'stockname','股票代码':'stockcode','日期':'trade_date','开盘':'open',\
                  '收盘':'close','最高':'high','最低':'low','成交量':'vol','成交额':'val','振幅':'amplitude',\
                      '涨跌幅':'change rate','涨跌额':'change','换手率':'turnover'}, inplace = True)
        d = stock['trade_date'].values
        for i in range(len(d)):
            d[i] = d[i][0:4] + d[i][5:7]+d[i][8:10]
        stock['vol'] = stock['vol'].astype(np.double)
        stock['closeMA90']=tl.MA(stock['close'].values, timeperiod=90)
        stock['closeMA30']=tl.MA(stock['close'].values, timeperiod=30)
        stock['closeMA15']=tl.MA(stock['close'].values, timeperiod=15)
        stock['volMA90']=tl.MA(stock['vol'].values, timeperiod=90)
        stock['volMA30']=tl.MA(stock['vol'].values, timeperiod=30)
        stock['volMA15']=tl.MA(stock['vol'].values, timeperiod=15)    
        stock['turnoverMA90'] = tl.MA(stock['turnover'].values,timeperiod = 90)
        stock['turnoverMA30'] = tl.MA(stock['turnover'].values,timeperiod = 30)
        stock['turnoverMA15'] = tl.MA(stock['turnover'].values,timeperiod = 15)
    else:
        if startdate[4:6] == '01':
            begdate = str(int(startdate)-8900)
        else:
            begdate = str(int(startdate)-100)
        sql = "select * from stockdata where stockcode='" + code + "' and trade_date>='"\
            +begdate+"' and trade_date<='"+enddate+"';"
        stock = pd.read_sql(sql,contrue)
    stock.dropna(axis=0, how='any', inplace=True)
    stock=stock.reset_index(drop=True)   
    startindex=stock[stock.trade_date>=startdate].index.tolist()
    if len(startindex) == 0:
        return None
    startposition=startindex[0]-window+1
    if startposition < 0:
        return None
    stock=stock.iloc[startposition:,:]
    return stock
    
    
cal_dates = pro.trade_cal()
today = datetime.date.today()
today=today.strftime('%Y%m%d')

while not is_trade_date(end):
    end=next_trade_day(end)

while not is_trade_date(start):
    start=next_day(start)
    if start > today:
        print('wrong test start date...')
        exit(0)
        
while not is_trade_date(end):
    end=next_day(end)
    if end > today:
        print('wrong test start date...')
        exit(0) 

watch_date = end
ct_days = 0
while next_trade_day(watch_date) < today:
    ct_days = ct_days + 1
    watch_date = next_trade_day(watch_date)
    if ct_days == extension:
        break

if next_trade_day(end) > today:
    showResult = False
    

modelname = model_dir + '\\Select16-16units10_4397.h5'

#os.chdir('f:\\stock')
basic = pd.read_pickle(data_dir + '\\Usualinfo.pkl')

def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def get_stock(code, startdate, enddate):
    or_stock=get_true_stock(code=code, startdate=startdate, enddate=enddate)
    #or_stock=ef.stock.get_quote_history(stock_codes=code, beg=str(int(startdate)-10000), end=enddate)
    ss = basic[basic['name']==code]
    maxP = ss.iloc[0]['highestPrice']
    maxV = ss.iloc[0]['maxVolumn']
    if or_stock is None:
        return None,None
    if len(or_stock) < window + ct_days:
        return None,None

    stock = or_stock[or_stock['trade_date'] <= end].copy()

    maxP = max(maxP,stock[['close','open','high','low']].values.max())
    maxV = max(maxV,stock['vol'].values.max())
    stock[['vol','volMA90','volMA30','volMA15']]=stock[['vol','volMA90','volMA30','volMA15']] * 0.99 / maxV
    stock[['close','open','high','low','closeMA90','closeMA30','closeMA15']] = \
        stock[['close','open','high','low','closeMA90','closeMA30','closeMA15']] * 0.99 / maxP
    stock[['turnover','turnoverMA90','turnoverMA30','turnoverMA15']] = \
        stock[['turnover','turnoverMA90','turnoverMA30','turnoverMA15']].abs().pow(0.5) / 10 
    stock['shares'] = np.sqrt(df[df['symbol']==code].iloc[0]['totalshare'] / max_shares) * 10

    return stock, or_stock

def create_dataset(code, data, or_data):
    global dateList, codeList, nameList, nextHigh, nextLow, nextOpen, nextClose, changeList, marketList
    label = 0
    x = []
    sample_len = len(data)
    stockname = basic[basic['name']==code].iloc[0].desc
    subset = data[['close','closeMA15','closeMA30','closeMA90','open','high','low',\
                         'vol','volMA15','volMA30','volMA90','turnover','turnoverMA15',\
                             'turnoverMA30','turnoverMA90','shares']].values
    for i in range(len(data) - window + 1):
        if  subset[i+window-1, label] == 0:
            continue
        else:
            dateList.append(data.iloc[i+window-1]['trade_date'])
            codeList.append('S '+code)
            nameList.append(stockname)
            closeList.append(or_data.iloc[i+window-1]['close'])
            marketList.append(df[df['symbol']==code].iloc[0]['market'])
            changeList.append(0)
            nextHigh.append(0)
            nextOpen.append(0)
            nextClose.append(0)
            nextLow.append(0)
            futureHigh.append(0)
            diff = extension if sample_len - i - window > extension else sample_len - i - window

            if diff > 1:
                fHigh = or_data.iloc[i+window+1:i+window+diff]['high'].values.max()
                changeList[-1] = (fHigh - closeList[-1]) / closeList[-1]
                nextHigh[-1] = or_data.iloc[i+window]['high']
                nextOpen[-1] = or_data.iloc[i+window]['open']
                nextClose[-1] = or_data.iloc[i+window]['close']
                nextLow[-1] = or_data.iloc[i+window]['low']
                futureHigh[-1] = fHigh

            x.append(subset[i:(i+window),:])
    return np.array(x)

init_gpu()
count = 0
print("Building dataset...\n")

for code in basic['name']:
    process = "\r[No:%4s %9s] "%(count,code)
    print(process, end='',flush=True)    
    stock, original = get_stock(code, start, watch_date)
    if stock is None:
        continue
    if len(stock) < window:
        continue
    xlist = create_dataset(code, stock, original)
    count = count + 1
    if len(trainX) == 0:
        trainX = xlist
    elif len(xlist) > 0:
        trainX = np.concatenate((trainX, xlist), axis = 0)
    else:
        continue

print("\n\nBuilding dataset completed\n")
print("start forecasting...\n")

outDF = pd.DataFrame()
outDF['code'] = codeList
outDF['market'] = marketList
outDF['date'] = dateList
outDF['name'] = nameList
outDF['close'] = closeList
outDF['totalchange'] = changeList
outDF['futureHigh'] = futureHigh
outDF['nextday open'] = nextOpen
outDF['nextday low'] = nextLow
outDF['nextday high'] = nextHigh
outDF['nextday close'] = nextClose

model=keras.models.load_model(modelname)
predict = model.predict(x=trainX,batch_size=batch_size)

outForecast = []


updowndict = {
    0: "不涨",
    1: "上涨",
}

def getUpdown(signal):
    return updowndict.get(signal, "Invalid signal")

for i in range(len(predict)):
    outForecast.append(getUpdown(np.argmax(predict[i])))  
    ratio.append(predict[i][1])
    
outDF['forecast'] = outForecast
outDF['ratio'] = ratio
print("Forecasting completed, saving result...\n")

print('\nCompleted verifying, save result\n')
        


outDF.to_csv(temp_dir+'\\forecastresult.csv',encoding='utf_8_sig')