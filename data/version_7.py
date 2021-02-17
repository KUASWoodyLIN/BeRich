import os
import numpy as np
from units.units import package_outputs


def load_data(stock_sql, load_dir=''):
    root_dir = os.path.join('dataset', os.path.basename(__file__).split('.')[0])
    stock_ids = stock_sql.read_1723_stock_ids()
    # 載入Stock ids
    if load_dir == '1723':
        train_stock_ids = set(stock_id[0] for stock_id in stock_ids)
    elif load_dir == '171':
        train_stock_ids = set(str(stock_id) for stock_id in stock_sql.read_171_stock_ids())
    else:
        load_dir = '171'
        train_stock_ids = set(str(stock_id) for stock_id in stock_sql.read_171_stock_ids())

    # Load Data
    x_train_file_1 = os.path.join(root_dir, load_dir, 'x_train_1.npy')
    x_train_file_2 = os.path.join(root_dir, load_dir, 'x_train_2.npy')
    x_train_file_3 = os.path.join(root_dir, load_dir, 'x_train_3.npy')
    x_train_file_4 = os.path.join(root_dir, load_dir, 'x_train_4.npy')
    x_train_file_5 = os.path.join(root_dir, load_dir, 'x_train_5.npy')
    x_train_file_6 = os.path.join(root_dir, load_dir, 'x_train_6.npy')
    x_val_file_1 = os.path.join(root_dir, load_dir, 'x_val_1.npy')
    x_val_file_2 = os.path.join(root_dir, load_dir, 'x_val_2.npy')
    x_val_file_3 = os.path.join(root_dir, load_dir, 'x_val_3.npy')
    x_val_file_4 = os.path.join(root_dir, load_dir, 'x_val_4.npy')
    x_val_file_5 = os.path.join(root_dir, load_dir, 'x_val_5.npy')
    x_val_file_6 = os.path.join(root_dir, load_dir, 'x_val_6.npy')
    y_train_file = os.path.join(root_dir, load_dir, 'y_train.npy')
    y_val_file = os.path.join(root_dir, load_dir, 'y_val.npy')
    if os.path.isfile(x_train_file_1) and os.path.isfile(x_train_file_2) and os.path.isfile(x_train_file_3) and \
            os.path.isfile(x_train_file_4) and os.path.isfile(x_train_file_5) and os.path.isfile(x_train_file_6) and \
            os.path.isfile(y_train_file) and os.path.isfile(x_val_file_1) and os.path.isfile(x_val_file_2) and \
            os.path.isfile(x_val_file_3) and os.path.isfile(x_val_file_4) and os.path.isfile(x_val_file_5) and \
            os.path.isfile(x_val_file_6) and os.path.isfile(y_val_file):
        x_train_1 = np.load(x_train_file_1)
        x_train_2 = np.load(x_train_file_2)
        x_train_3 = np.load(x_train_file_3)
        x_train_4 = np.load(x_train_file_4)
        x_train_5 = np.load(x_train_file_5)
        x_train_6 = np.load(x_train_file_6)
        x_val_1 = np.load(x_val_file_1)
        x_val_2 = np.load(x_val_file_2)
        x_val_3 = np.load(x_val_file_3)
        x_val_4 = np.load(x_val_file_4)
        x_val_5 = np.load(x_val_file_5)
        x_val_6 = np.load(x_val_file_6)
        y_train = np.load(y_train_file)
        y_val = np.load(y_val_file)
        print("Finish data loading")
    else:
        # Load Data
        y_train = []
        y_val = []
        x_train_1 = []
        x_train_2 = []
        x_train_3 = []
        x_train_4 = []
        x_train_5 = []
        x_train_6 = []
        x_val_1 = []
        x_val_2 = []
        x_val_3 = []
        x_val_4 = []
        x_val_5 = []
        x_val_6 = []
        mean_dict_1 = {}
        mean_dict_2 = {}
        mean_dict_3 = {}
        mean_dict_4 = {}
        mean_dict_5 = {}
        std_dict_1 = {}
        std_dict_2 = {}
        std_dict_3 = {}
        std_dict_4 = {}
        std_dict_5 = {}
        for stock_id in stock_ids:
            if type(stock_id) in [tuple, list]:
                stock_id = stock_id[0]
            # 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量
            select = "SELECT 日期, 開, 收, MA10, MA20, MA60, K9, D9, OSC, \
                      收, UB20, PB, BW, \
                      成交量, 成交量'(流通比)', MV5,  \
                      收, wMA20, wMACD9, wOSC,  \
                      外資3, 外資5, 外資10, 外資20, 外資60 \
                      FROM 技術面__stockall \
                      WHERE 代號 = '" + str(stock_id) + "' \
                      AND 日期 <= '2019-12-31' \
                      AND 日期 >= '2006-01-01' \
                      ORDER BY 日期 DESC \
                      LIMIT 4000 \
                      ;"
            values = list(stock_sql.read_stock_values(select))
            # 移除 MA20, K9, D9, OSC, UB20, PB, BW, 外資... 為NULL的資料
            values = [v for v in values if 0 not in v[3:-1] and None not in v[3:-1]]
            if len(values) < 2000:
                print('(False) Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
                continue

            # Create X Input 1
            x_data_1 = np.array(values)[:, 1:9].astype(np.float)
            mean_1 = x_data_1.mean(axis=0)
            std_1 = x_data_1.std(axis=0)
            x_data_1 = (x_data_1[30:] - mean_1) / std_1
            mean_dict_1[stock_id] = mean_1
            std_dict_1[stock_id] = std_1
            # Create X Input 2
            x_data_2 = np.array(values)[:, 9:13].astype(np.float)
            mean_2 = x_data_2.mean(axis=0)
            std_2 = x_data_2.std(axis=0)
            x_data_2 = (x_data_2[30:] - mean_2) / std_2
            mean_dict_2[stock_id] = mean_2
            std_dict_2[stock_id] = std_2
            # Create X Input 3
            x_data_3 = np.array(values)[:, 13:16].astype(np.float)
            mean_3 = x_data_3.mean(axis=0)
            std_3 = x_data_3.std(axis=0)
            x_data_3 = (x_data_3[30:] - mean_3) / std_3
            mean_dict_3[stock_id] = mean_3
            std_dict_3[stock_id] = std_3
            # Create X Input 4
            x_data_4 = np.array(values)[:, 16:20].astype(np.float)
            mean_4 = x_data_4.mean(axis=0)
            std_4 = x_data_4.std(axis=0)
            x_data_4 = (x_data_4[30:] - mean_4) / std_4
            mean_dict_4[stock_id] = mean_4
            std_dict_4[stock_id] = std_4
            # Create X Input 5
            x_data_5 = np.array(values)[:, 20:].astype(np.float)
            mean_5 = x_data_5.mean(axis=0)
            std_5 = x_data_5.std(axis=0)
            x_data_5 = (x_data_5[30:] - mean_5) / std_5
            mean_dict_5[stock_id] = mean_5
            std_dict_5[stock_id] = std_5
            # Create X Input 6 (day1~5 (收-開)/開)
            open_prices = np.array(values)[:, 1].astype(np.float)
            close_prices = np.array(values)[:, 2].astype(np.float)
            increase_p = (close_prices - open_prices) / open_prices
            increase_p = np.append(increase_p, [0, 0, 0, 0])
            x_data_6 = np.array([increase_p[idx:idx+5] for idx in range(len(increase_p)-4)]).astype(np.float)[30:]

            # Create Y data
            y_data = []
            for idx in range(30, len(values), 1):
                if values[idx][2] * 1.2 < max([v[2] for v in values[idx-30:idx]]):
                    y_data.append(1)
                else:
                    y_data.append(0)

            # Find training & valuation split point
            split_point = 0
            dates = [str(v[0]) for v in values[30:]]
            for i, j in enumerate(dates):
                if j == '2016-12-30':
                    split_point = i

            # Split data into train/val
            if stock_id in train_stock_ids:
                x_train_1.append(x_data_1[split_point:])
                x_train_2.append(x_data_2[split_point:])
                x_train_3.append(x_data_3[split_point:])
                x_train_4.append(x_data_4[split_point:])
                x_train_5.append(x_data_5[split_point:])
                x_train_6.append(x_data_6[split_point:])
                y_train += y_data[split_point:]
            x_val_1.append(x_data_1[:split_point])
            x_val_2.append(x_data_2[:split_point])
            x_val_3.append(x_data_3[:split_point])
            x_val_4.append(x_data_4[:split_point])
            x_val_5.append(x_data_5[:split_point])
            x_val_6.append(x_data_6[:split_point])
            y_val += y_data[:split_point]
            print('Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))

        # Combine all stocks
        x_train_1 = np.concatenate(x_train_1)
        x_train_2 = np.concatenate(x_train_2)
        x_train_3 = np.concatenate(x_train_3)
        x_train_4 = np.concatenate(x_train_4)
        x_train_5 = np.concatenate(x_train_5)
        x_train_6 = np.concatenate(x_train_6)
        x_val_1 = np.concatenate(x_val_1)
        x_val_2 = np.concatenate(x_val_2)
        x_val_3 = np.concatenate(x_val_3)
        x_val_4 = np.concatenate(x_val_4)
        x_val_5 = np.concatenate(x_val_5)
        x_val_6 = np.concatenate(x_val_6)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # Save data at local
        if not os.path.isdir(os.path.join(root_dir, load_dir)):
            os.makedirs(os.path.join(root_dir, load_dir))
        np.save(os.path.join(root_dir, load_dir, 'mean_1.npy'), mean_dict_1)
        np.save(os.path.join(root_dir, load_dir, 'mean_2.npy'), mean_dict_2)
        np.save(os.path.join(root_dir, load_dir, 'mean_3.npy'), mean_dict_3)
        np.save(os.path.join(root_dir, load_dir, 'mean_4.npy'), mean_dict_4)
        np.save(os.path.join(root_dir, load_dir, 'mean_5.npy'), mean_dict_5)
        np.save(os.path.join(root_dir, load_dir, 'std_1.npy'), std_dict_1)
        np.save(os.path.join(root_dir, load_dir, 'std_2.npy'), std_dict_2)
        np.save(os.path.join(root_dir, load_dir, 'std_3.npy'), std_dict_3)
        np.save(os.path.join(root_dir, load_dir, 'std_4.npy'), std_dict_4)
        np.save(os.path.join(root_dir, load_dir, 'std_5.npy'), std_dict_5)
        np.save(os.path.join(root_dir, load_dir, 'x_train_1.npy'), x_train_1)
        np.save(os.path.join(root_dir, load_dir, 'x_train_2.npy'), x_train_2)
        np.save(os.path.join(root_dir, load_dir, 'x_train_3.npy'), x_train_3)
        np.save(os.path.join(root_dir, load_dir, 'x_train_4.npy'), x_train_4)
        np.save(os.path.join(root_dir, load_dir, 'x_train_5.npy'), x_train_5)
        np.save(os.path.join(root_dir, load_dir, 'x_train_6.npy'), x_train_6)
        np.save(os.path.join(root_dir, load_dir, 'x_val_1.npy'), x_val_1)
        np.save(os.path.join(root_dir, load_dir, 'x_val_2.npy'), x_val_2)
        np.save(os.path.join(root_dir, load_dir, 'x_val_3.npy'), x_val_3)
        np.save(os.path.join(root_dir, load_dir, 'x_val_4.npy'), x_val_4)
        np.save(os.path.join(root_dir, load_dir, 'x_val_5.npy'), x_val_5)
        np.save(os.path.join(root_dir, load_dir, 'x_val_6.npy'), x_val_6)
        np.save(os.path.join(root_dir, load_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(root_dir, load_dir, 'y_val.npy'), y_val)
        print("Finish data loading")
    x_train = (x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6)
    x_val = (x_val_1, x_val_2, x_val_3, x_val_4, x_val_5, x_val_6)
    return x_train, y_train, x_val, y_val


def load_and_pred_data(stock_sql, model):
    ret = []
    root_dir = os.path.join('dataset', os.path.basename(__file__).split('.')[0])
    version = int(os.path.basename(__file__).split('.')[0].split('_')[-1])

    mean_dict_1 = np.load(os.path.join(root_dir, '1723', 'mean_1.npy'), allow_pickle=True).item()
    mean_dict_2 = np.load(os.path.join(root_dir, '1723', 'mean_2.npy'), allow_pickle=True).item()
    mean_dict_3 = np.load(os.path.join(root_dir, '1723', 'mean_3.npy'), allow_pickle=True).item()
    mean_dict_4 = np.load(os.path.join(root_dir, '1723', 'mean_4.npy'), allow_pickle=True).item()
    mean_dict_5 = np.load(os.path.join(root_dir, '1723', 'mean_5.npy'), allow_pickle=True).item()
    std_dict_1 = np.load(os.path.join(root_dir, '1723', 'std_1.npy'), allow_pickle=True).item()
    std_dict_2 = np.load(os.path.join(root_dir, '1723', 'std_2.npy'), allow_pickle=True).item()
    std_dict_3 = np.load(os.path.join(root_dir, '1723', 'std_3.npy'), allow_pickle=True).item()
    std_dict_4 = np.load(os.path.join(root_dir, '1723', 'std_4.npy'), allow_pickle=True).item()
    std_dict_5 = np.load(os.path.join(root_dir, '1723', 'std_5.npy'), allow_pickle=True).item()
    # Run all dates
    for stock_id in mean_dict_1.keys():
        select = "SELECT 日期, 開, 收, MA10, MA20, MA60, K9, D9, OSC, \
                  收, UB20, PB, BW, \
                  成交量, 成交量'(流通比)', MV5,  \
                  收, wMA20, wMACD9, wOSC,  \
                  外資3, 外資5, 外資10, 外資20, 外資60 \
                  FROM 技術面__stockall \
                  WHERE 代號 = '" + str(stock_id) + "' \
                  AND 日期 >= '2006-01-01' \
                  ORDER BY 日期 DESC \
                  LIMIT 4000 \
                  ;"

        values = list(stock_sql.read_stock_values(select))
        # 移除 MA20, K9, D9, OSC, UB20, PB, BW, 外資... 為NULL的資料
        values = [v for v in values if 0 not in v[3:-1] and None not in v[3:-1]]
        if len(values) < 1:
            print('(False) Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
            continue

        # Create X Input 1
        x_data_1 = np.array(values)[:, 1:9].astype(np.float)
        x_data_1 = (x_data_1 - mean_dict_1[stock_id]) / std_dict_1[stock_id]

        # Create X Input 2
        x_data_2 = np.array(values)[:, 9:13].astype(np.float)
        x_data_2 = (x_data_2 - mean_dict_2[stock_id]) / std_dict_2[stock_id]

        # Create X Input 3
        x_data_3 = np.array(values)[:, 13:16].astype(np.float)
        x_data_3 = (x_data_3 - mean_dict_3[stock_id]) / std_dict_3[stock_id]

        # Create X Input 4
        x_data_4 = np.array(values)[:, 16:20].astype(np.float)
        x_data_4 = (x_data_4 - mean_dict_4[stock_id]) / std_dict_4[stock_id]

        # Create X Input 5
        x_data_5 = np.array(values)[:, 20:].astype(np.float)
        x_data_5 = (x_data_5 - mean_dict_5[stock_id]) / std_dict_5[stock_id]

        # Predict
        outputs = model.predict((x_data_1, x_data_2, x_data_3, x_data_4, x_data_5))

        # Save predict result
        dates = [str(v[0]) for v in values]
        ret += package_outputs(stock_id, dates, outputs, version)
        print('Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
    return ret


if __name__ == '__main__':
    from units.sql import StockSQL

    # 連線SQL資料庫
    stock_sql_test = StockSQL()

    # 載入資料
    load_data(stock_sql_test, load_dir='1723')
