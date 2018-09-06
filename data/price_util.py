import numpy as np
import pandas as pd
import struct
from gc import collect
import sqlite3

class Config:
    def __init__(self):
        self.rawdata_path = '/data2/stock/NYSE/original/raw/'
        self.index_path = '/home/duxin/code/FinancialDataResearch/index/'
        self.table1 = pd.read_csv(self.index_path + '/table1.csv', header = None).set_index([0, 1])
        self.table2 = pd.read_csv(self.index_path + '/table2.csv').set_index('StockName')
        self.table3 = pd.read_csv(self.index_path + '/table3.csv').set_index('FileID')

config = Config()

def _GetTradingDayList(stock_id, start_time, end_time):
    date_list = pd.date_range(start = start_time, end = end_time)
    date_list = [date.year*10**4 + date.month*10**2 + date.day for date in date_list]
    trading_day_list = [date for date in date_list if (date, stock_id) in config.table1.index]
    return trading_day_list

def _GetSinglePrice(conn, stock_id, day, sample_moment):
    cur = conn.cursor()
    start_time_of_day = pd.to_datetime(str(day)).value
    end_time_of_day = start_time_of_day + 86400000000000 - 1
    if sample_moment == 'end':
        cmd = 'select * from secondaryindex where StockId = %d and datetime Between %d and %d ORDER by datetime DESC limit 1'%(stock_id, start_time_of_day, end_time_of_day)
    record = cur.execute(cmd).fetchall()
    if len(record) == 0:
        print('found no records on %d'%day)
        return None
    else:
        record = record[0]
    
    with open(config.index_path + 'table4.%04d.%02d.bin.modified'%(int(day/10000), int(day%10000/100)), 'rb') as f:
        f.seek(record[2], 0)
        datetime, file_id, stock_id, first_length, last_length, offset, chunk_length = struct.unpack('lHHHHll', f.read(32))
    
    file_name = config.table3.loc[file_id][0]
    with open(config.rawdata_path + file_name, 'rt') as f:
        f.seek(offset + chunk_length - last_length, 0)
        price = float(f.read(last_length).split(',')[7])
    return (datetime, price)

def GetStockPriceByDays(stock_name, start_time, end_time, interval = 1, sample_moment = 'end', as_dataframe = False):
    stock_id = config.table2.loc[stock_name][0]
    trading_day_list = _GetTradingDayList(stock_id, start_time, end_time)
    sample_day_list = trading_day_list[0::interval]
    conn = sqlite3.Connection(config.index_path + 'secondaryindex.db')
    price_hist = []
    num_sample = len(sample_day_list)
    for i in range(num_sample):
        if i%10 == 0 or i == num_sample - 1:
            print('progress %d / %d'%(i+1, num_sample))
        price = _GetSinglePrice(conn, stock_id, sample_day_list[i], sample_moment)
        price_hist.append(price)
    conn.close()
    if as_dataframe == True:
        price_hist = pd.DataFrame(price_hist, columns = ['datetime', 'price'])
        price_hist['datetime'] = pd.to_datetime(price_hist['datetime'])
    return price_hist

def GetStockPricesWithinDay(stock_name, start_time, end_time, interval = 60, sample_moment = 'end', as_dataframe = False):
    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)
    if start_datetime.year != end_datetime.year or start_datetime.month != end_datetime.month or start_datetime.day != end_datetime.day:
        raise ValueError('"start_time" and "end_time" must be in the same year/month/day.')
        return None
    year = start_datetime.year
    month = start_datetime.month
    day = start_datetime.day
    stock_id = config.table2.loc[stock_name][0]
    date = '%04d%02d%02d'%(year, month, day)
    start_time_of_day = pd.to_datetime(date).value
    end_time_of_day = start_time_of_day + 86400000000000 - 1
    conn = sqlite3.Connection(config.index_path + 'secondaryindex.db')
    cur = conn.cursor()
    data = cur.execute('select * from secondaryindex where StockId = %d and datetime Between %d and %d ORDER by datetime'%(stock_id, start_time_of_day, end_time_of_day)).fetchall()
    if len(data) == 0:
        print('found no records on %s'%date)
        return None
    else:
        earliest_record = data[0]
        earliest_record_datetime = earliest_record[0]
        latest_record = data[-1]
        latest_record_datetime = latest_record[0]
    sample_time_list = pd.to_numeric(pd.date_range(start = start_time, end = end_time, freq = '%dS'%interval))
    sample_time_list = sample_time_list[(sample_time_list >= earliest_record_datetime) & (sample_time_list <= latest_record_datetime)]
    if len(sample_time_list) == 0:
        print('found no records in the specified time period')
    actual_time_list = []
    i = 0
    num_data = len(data)
    for sample in sample_time_list:
        while i < num_data and sample > data[i][0]:
            i += 1
        actual_time_list.append(data[i])
    actual_time_list = sorted(list(set(actual_time_list)))

    index_pos = []
    with open(config.index_path + 'table4.%04d.%02d.bin.modified'%(year, month), 'rb') as f:
        for sample in actual_time_list:
            f.seek(sample[2], 0)
            index_pos.append(struct.unpack('lHHHHll', f.read(32)))
    file_id = index_pos[0][1]
    file_name = config.table3.loc[file_id][0]

    price_hist = []
    progress = 0
    num_sample = len(index_pos)
    with open(config.rawdata_path + file_name, 'rt') as f:
        for sample in index_pos:
            progress += 1
            if (progress % 100 == 0) or (progress == num_sample - 1):
                print('progress %d / %d.'%(progress + 1, num_sample))
            f.seek(sample[5] + sample[6] - sample[4], 0)
            record = f.read(sample[4]).split(',')
            price_hist.append((sample[0], float(record[7])))
    price_hist = pd.DataFrame(price_hist, columns = ['datetime', 'price'])
    
    if as_dataframe == True:
        price_hist['datetime'] = pd.to_datetime(price_hist['datetime'])

    return price_hist

def GetStockPriceOverDays(stock_name, start_time, end_time, interval = 60):
    # only day-level timestamp is supported
    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)
    day_list = pd.date_range(start_datetime, end_datetime, freq = '1d')
    price_hist = pd.DataFrame(columns = ['datetime', 'price'])
    # iteratively retrieve daily price
    for day in day_list:
        print("Extracting day-level price data of {}".format(day))
        price = GetStockPricesWithinDay(stock_name, 
                    start_time = day.replace(hour=0,minute=0,second=0).strftime('%Y%m%d %H:%M:%S'),
                    end_time = day.replace(hour=23,minute=59,second=59).strftime('%Y%m%d %H:%M:%S'),
                    interval = interval)
        price_hist = pd.concat([price_hist, price], axis = 0)
    return price_hist