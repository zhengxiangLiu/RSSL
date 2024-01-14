# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:44:27 2021

@author: DELL
"""

# -*- coding: utf-8 -*-

from collections import deque
import networkx as nx
import json
import numpy as np
import os
import argparse
import pickle
import matplotlib.pyplot as plt


class SectorPreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file,
                                 selected_tickers_fname,
                                 connection_file, tic_wiki_file,
                                 sel_path_file):
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path, selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print(selected_tickers[0:5])
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))
        G2 = nx.Graph()
        for industry in industry_tickers:
            for i in range(len(industry_tickers[industry])):
                stock1 = ticker_index[industry_tickers[industry][i]]
                G2.add_node(stock1)
                for j in range(i + 1, len(industry_tickers[industry])):
                    stock2 = ticker_index[industry_tickers[industry][j]]
                    G2.add_edge(stock1, stock2)

        valid_industry_count = 0
        valid_industry_index = {}
        for industry in industry_tickers.keys():
            if len(industry_tickers[industry]) > 1:
                valid_industry_index[industry] = valid_industry_count
                valid_industry_count += 1
        ticker_relation_embedding = np.zeros([len(selected_tickers), valid_industry_count + 1], dtype=int)
        one_hot_industry_embedding = np.identity(valid_industry_count + 1, dtype=int)
        print("#ticker_relation_embedding:", ticker_relation_embedding.shape)
        print("#one_hot_industry_embedding:", one_hot_industry_embedding.shape)
        trel = []
        ttar = []
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]
            if len(cur_ind_tickers) <= 1:
                print('shit industry:', industry)
                continue
            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                rel = []
                tar = []
                right_tic_ind = ticker_index[cur_ind_tickers[i]]
                ttar.append(right_tic_ind)
                for j in range(len(cur_ind_tickers)):
                    left_tic_ind = ticker_index[cur_ind_tickers[j]]
                    if i != j:
                        rel.append(left_tic_ind)
                trel.append(rel)

        tickers = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',',
                                skip_header=False)
        print('#tickers selected:', tickers.shape)
        wikiid_ticind_dic = {}
        for ind, tw in enumerate(tickers):
            if not tw[-1] == 'unknown':
                wikiid_ticind_dic[tw[-1]] = ind
        print('#tickers aligned:', len(wikiid_ticind_dic))
        sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
                                  skip_header=False)
        print('#paths selected:', len(sel_paths))
        sel_paths = set(sel_paths[:, 0])
        with open(connection_file, 'r') as fin:
            connections = json.load(fin)
        print('#connection items:', len(connections))
        occur_paths = set()
        for sou_item, conns in connections.items():
            for tar_item, paths in conns.items():
                for p in paths:
                    path_key = '_'.join(p)
                    if path_key in sel_paths:
                        occur_paths.add(path_key)
        valid_path_index = {}
        for ind, path in enumerate(occur_paths):
            valid_path_index[path] = ind
        print('#valid paths:', len(valid_path_index))
        for path, ind in valid_path_index.items():
            print(path, ind)
        wiki_relation_embedding = np.zeros(
            [tickers.shape[0], len(valid_path_index) + 1],
            dtype=int
        )
        conn_count = 0
        # for path_key in valid_industry_index.keys():
        # ccc = valid_path_index[path_key]
        flag = False
        sorse_stock = []
        end_stock = []
        for sou_item, conns in connections.items():
            rel = []
            aaa = wikiid_ticind_dic[sou_item]
            for tar_item, paths in conns.items():
                if tar_item is not None:
                    for p in paths:
                        path_key = '_'.join(p)
                        # print(path_key)
                        if path_key in valid_path_index.keys():
                            sorse_stock.append(sou_item)
                            end_stock.append(tar_item)
                            bbb = wikiid_ticind_dic[tar_item]
                            rel.append(bbb)
                            flag = True
            if flag:
                trel.append(rel)
                ttar.append(aaa)
                flag = False
        # trel=np.asarray(trel)
        # ttar=np.asarray(ttar)
        # sorse_stock=set(sorse_stock)
        # end_stock=set(end_stock)
        # G2 = nx.Graph()
        for stock1, stock2 in zip(sorse_stock, end_stock):
            G2.add_edge(wikiid_ticind_dic[stock1], wikiid_ticind_dic[stock2])
        tree2s = []
        max_deep=3
        sorse_stock2 = set(sorse_stock)
        for root_node in selected_tickers:
            root_node = ticker_index[root_node]
            tree = {root_node: []}
            has_on = [root_node]
            visited = set()
            queue = deque([root_node])
            queue2 = deque()
            for i in range(max_deep-1):
                while queue:
                    node = queue.popleft()
                    visited.add(node)
                    for neighbor in G2.neighbors(node):
                        if neighbor not in visited and neighbor not in has_on:
                            has_on.append(neighbor)
                            queue2.append(neighbor)
                            tree[node].append(neighbor)
                            tree[neighbor] = []
                queue=queue2
                queue2 = deque()
            tree2s.append(tree)

        with open('../data/tree/NASDAQ_tree_File.txt', 'wb') as fw:
            pickle.dump(tuple(tree2s), fw)



if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = '../data/'
    if args.market is None:
        args.market = 'NASDAQ'
        # args.market = 'NYSE'

    processor = SectorPreprocessor(args.path, args.market)
    path = '../data/relation/wikidata/'
    processor.generate_sector_relation(
        os.path.join('../data/relation/sector_industry/',
                     processor.market_name + '_industry_ticker.json'),
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv',
        os.path.join(path, 'NYSE_connections.json'),
        os.path.join(path, 'NYSE_wiki.csv'),
        os.path.join(path, 'selected_wiki_connections.csv')
    )