
import json
import numpy as np
import os


def build_wiki_relation(market_name, connection_file, tic_wiki_file,
                        sel_path_file):
    # readin tickers
    tickers = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',',
                            skip_header=False)
    print('#tickers selected:', tickers.shape)
    wikiid_ticind_dic = {}
    for ind, tw in enumerate(tickers):
        if not tw[-1] == 'unknown':
            wikiid_ticind_dic[tw[-1]] = ind

    print('#tickers aligned:', len(wikiid_ticind_dic))

    # readin selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
                              skip_header=False)
    print('#paths selected:', len(sel_paths))
    sel_paths = set(sel_paths[:, 0])

    # readin connections
    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
    print('#connection items:', len(connections))

    # get occured paths
    occur_paths = set()
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in sel_paths:
                    occur_paths.add(path_key)

    # generate
    valid_path_index = {}
    for ind, path in enumerate(occur_paths):
        valid_path_index[path] = ind
    print('#valid paths:', len(valid_path_index))
    for path, ind in valid_path_index.items():
        print(path, ind)
    # one_hot_path_embedding = np.identity(len(valid_path_index) + 1, dtype=int)
    wiki_relation_embedding = np.zeros(
        [tickers.shape[0], len(valid_path_index) + 1],
        dtype=int
    )
    one_hot_embedding = np.identity(len(valid_path_index) + 1, dtype=float)
    conn_count = 0
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in valid_path_index.keys():
                    aaa = wikiid_ticind_dic[sou_item]
                    bbb = wikiid_ticind_dic[tar_item]
                    ccc = valid_path_index[path_key]
                    wiki_relation_embedding[wikiid_ticind_dic[sou_item]][valid_path_index[path_key]] = -1
                    wiki_relation_embedding[wikiid_ticind_dic[tar_item]][valid_path_index[path_key]] = 1

                    conn_count += 1
    print('connections count:', conn_count, 'ratio:', conn_count / float(tickers.shape[0] * tickers.shape[0]))
    for i in range(0, len(valid_path_index) + 1):
        for j in range(i + 1, len(valid_path_index) + 1):
            count = float(0)
            for k in range(0, wiki_relation_embedding.shape[0]):
                if (abs(wiki_relation_embedding[k][i]) == abs(wiki_relation_embedding[k][i])):
                    count += 1
            one_hot_embedding[i][j] = 1 / count
            one_hot_embedding[j][i] = 1 / count

    np.save(market_name + '_wiki_relation', wiki_relation_embedding)
    np.save(market_name + '_wiki_relation_edge', one_hot_embedding)


# single thread version
if __name__ == '__main__':
    path = 'D:/Graduation_data/Gcoder/data/relation/wikidata/'
    build_wiki_relation('NASDAQ',
                        os.path.join(path, 'NASDAQ_connections.json'),
                        os.path.join(path, 'NASDAQ_wiki.csv'),
                        os.path.join(path, 'selected_wiki_connections.csv'))
    print('----------')
    build_wiki_relation('NYSE',
                        os.path.join(path, 'NYSE_connections.json'),
                        os.path.join(path, 'NYSE_wiki.csv'),
                        os.path.join(path, 'selected_wiki_connections.csv'))