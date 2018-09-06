# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from price_util import *
import pandas as pd
import logging
import time
import multiprocessing as mtp

logger = logging.getLogger('main_logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(filename = 'price_data_extraction.log', mode = 'a'))
logger.addHandler(logging.StreamHandler())

interval = 300 # 300 seconds

def ExtractOneStockPriceHist(stock_name):
    t0 = time.time()
    logger.info("STOCK '{}' started extracting. Current time: {}."
                    .format(stock_name, time.ctime()))
    price_hist = GetStockPriceOverDays(stock_name, '20150101', '20171231', interval=interval)
    logger.info("STOCK '{}' finished extracting. Current time: {}. Time consumed: {} seconds."
                    .format(stock_name, time.ctime(), time.time() - t0))
    price_hist.to_csv('./prices.5min/{}.5min.csv'.format(stock_name), index = None)

n_process = 50
processes = []
p = mtp.Pool(n_process)
stock_names = set(config.table2.index.tolist())
selected_stock_names = pd.read_csv('top100volume_names.csv', header = None).values.squeeze().tolist()
for stock_name in selected_stock_names:
    if stock_name not in stock_names:
        selected_stock_names.remove(stock_name)
        logger.error("Stock '{}' exists not.".format(stock_name))
p.map(ExtractOneStockPriceHist, selected_stock_names[:50])

