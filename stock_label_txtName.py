import os  #通过os模块调用系统命令
import pandas as pd
import numpy as np

def company_transcripts(file):
    file_path = "earning_call/" + file + "/"  # 文件路径
    path_list = os.listdir(file_path)  # 遍历整个文件夹下的文件name并返回一个列表
    path_name = []  # 定义一个空列表
    for i in path_list:
        path_name.append(i.split(".")[0])  # 若带有后缀名，利用循环遍历path_list列表，split去掉后缀名
    path_name.sort()  # 排序
    path_name.pop(0)
    for transcript in path_name:
        detail = {}
        x = transcript.split('-')
        x.reverse()
        x[2] = months[x[2]]
        symbol = x.pop(0)
        date = '-'.join(x)
        detail['Company'] = symbol
        detail['Date'] = date
        #print(symbol)
        #print(date)
        stock_file = 'stock_price/' + symbol + '.csv'
        result = stock_calculating(stock_file, date)
        detail['Day1'] = result[0]
        detail['Day5'] = result[1]
        detail['Transcript_path'] = transcript + ".txt"
        cont_list.append(detail)
def stock_calculating(path, date):
    results = []
    row1 = 0
    row2 = 0
    data = pd.read_csv(path)
    for i in range(len(data['Date'])):
        if data['Date'][i] == date:
            row1 = i
            break
    stock_change1 = (data.iloc[row1+1]['Close'] - data.iloc[row1]['Close']) / data.iloc[row1]['Close']
    stock_change5 = (data.iloc[row1+5]['Close'] - data.iloc[row1]['Close']) / data.iloc[row1]['Close']
    ndaq = pd.read_csv('NASDAQ_17_22.csv')
    for i in range(len(ndaq['Date'])):
        if ndaq['Date'][i] == date:
            row2 = i
            break
    ndaq_change1 = (ndaq.iloc[row2 + 1]['Close-Last'] - ndaq.iloc[row2]['Close-Last']) / ndaq.iloc[row2]['Close-Last']
    ndaq_change5 = (ndaq.iloc[row2 + 5]['Close-Last'] - ndaq.iloc[row2]['Close-Last']) / ndaq.iloc[row2]['Close-Last']
    if stock_change1 > 0:
        if ndaq_change1 < 0:
            results.append(1)
        elif ndaq_change1 > 0:
            results.append(1) if stock_change1 - ndaq_change1 > 0 else results.append(0)
    else:
        results.append(0)
    if stock_change5 > 0:
        if ndaq_change5 < 0:
            results.append(1)
        elif ndaq_change5 > 0:
            results.append(1) if stock_change5 - ndaq_change5 > 0 else results.append(0)
    else:
        results.append(0)
    return results
if __name__ == '__main__':
    # 将月份都改为字符数字
    months = {'Jan': '01', 'Feb': '02',
              'Mar': '03', 'Apr': '04',
              'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08',
              'Sep': '09', 'Oct': '10',
              'Nov': '11', 'Dec': '12', }
    files = ['AAPL', 'AMGN', 'AMZN', 'COST', 'MSFT', 'MU']
    cont_list = []
    for file in files:
        company_transcripts(file)
    df = pd.DataFrame(cont_list, columns=["Company", "Date", "Day1", "Day5","Transcript_path"])
    print(df)
    df.to_csv("trans_stock_union.csv", index="False")