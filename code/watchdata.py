# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:33:41 2022

@author: kenneth
"""
import datetime
import pandas as pd
from tkinter import *
import tushare as ts
import os
import efinance as ef

root_dir = 'f:git-pro\\play-stock'
code_dir = root_dir + '\\code'
data_dir = root_dir + '\\data'
model_dir = root_dir + '\\model'
temp_dir = root_dir + '\\temp'

filename = temp_dir + 'newestforecast.csv'
stocklist = []
codelist = []
stockno = 0
count = 0

predict_stock = pd.read_csv(filename)

datelist = predict_stock.trade_date
datelist = datelist.sort_values(ascending=False, inplace=False)
datelist = datelist[~datelist.duplicated()].reset_index()

# os.chdir('d:\\python\\stock')
ts.set_token('7b571c7a6118274b004dbdddde8c48a8ed5a656803e441584c4a5053')
pro = ts.pro_api()

today = datetime.date.today()
today=today.strftime('%Y%m%d')
cal_dates = pro.trade_cal()


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

def update_price(allstock):
    price = []
    for code in codelist:
        stock_code = code[0:6]
        stockdf = ef.stock.get_quote_history(stock_code, klt=1)
        if len(stockdf) > 0:
            price.append(stockdf.iloc[-1]['收盘'])
        else:
            price.append(0)
    return price

        

days = (len(datelist),5)[len(datelist)>5]
datelist = datelist[0:days]['trade_date'].tolist()

area = []
daytitle = '            '
for i in range(0,days):
    daytitle = daytitle + str(datelist[i]) + '    ' 

def to12(original):
    add = 12 - len(original)
    tp = original + ' '*add
    return tp

area=[['' for i in range(days)] for j in range(100)]
buylist=[[0 for i in range(days)] for j in range(100)]        
col = 0
for date in datelist:
    one_day_stock = predict_stock[predict_stock['trade_date']==date]
    for index, row in one_day_stock.iterrows():
        name = row['stockname']
        code = row['stockcode']
        if not (code in codelist): 
            stocklist.append(name)
            codelist.append(code)
            stockno = stockno + 1
        idx = codelist.index(code)
        area[idx][col] = name
        buylist[idx][col] = '%.2f'%row ['buy']
    col = col + 1

for i in range(col):
    datelist[i] = str(datelist[i])
    for j in range(stockno):
        if buylist[j][i] == 0:
            buylist[j][i] = ''

window = Tk()
window.title('Stock 5 days table')


def set_clock():
    def clock():
        global count
        count = count + 1
        time = datetime.datetime.strftime(datetime.datetime.now(),'%H:%M:%S')
        timeLabel.config(text=time)
        if count % 10 == 0:
            count = 0
            new_price = update_price(codelist)
            for i in range(stockno):
                for j in range(col):
                    color = 'red'
                    if buylist[i][j] == '':
                        percent =''
                    else:
                        buy = float(buylist[i][j])
                        change = '%.2f'%(100*(new_price[i]-buy)/buy)
                        if new_price[i] < buy:
                            color = 'green'
                        if new_price[i] == 0:
                            color = 'black'
                            change = ' '
                        else:
                            if len(change) > 6:
                                change = change[0:6]
                        percent = buylist[i][j] + '  ' + change + '%'
                    buyLabel[i][j].config(text = percent, fg = color)
        timeLabel.after(1000, clock)
    clock()

timeLabel = Label(window, text='Time', fg = 'red', width=20, height=2, font=('Helvetic', 15, 'bold'))
timeLabel.grid(row=0,column=3)


dateLabel = [Label(window, text=date, fg='blue', width=20, height=2, font=('Helvetic', 15, 'bold')) for date in datelist]
for i in range(0,col):
    dateLabel[i].grid(row=1, column=i+1)

stockLabel = [Label(window, text=stock, bg='grey', fg='purple', width=20, height=1, font=('Helvetic', 10, 'bold')) for stock in stocklist]
for i in range(0,stockno):
    stockLabel[i].grid(row=i+2,column=0)

buyLabel = []
for i in range(stockno):
    lb_line = []
    if i%2:
        bgcl = 'grey'
    else:
        bgcl = 'lightgrey'
    stockLabel[i].config(bg=bgcl)
    for j in range(col):
        lb_line.append(Label(window, text=buylist[i][j], bg = bgcl, width=20, height=1, font=('Helvetic', 10, 'bold')))
        lb_line[j].grid(row=i+2,column=j+1)
    buyLabel.append(lb_line)


set_clock()
window.mainloop()

