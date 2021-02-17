import os
import numpy as np


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
    x_val_file_1 = os.path.join(root_dir, load_dir, 'x_val_1.npy')
    x_val_file_2 = os.path.join(root_dir, load_dir, 'x_val_2.npy')
    y_train_file = os.path.join(root_dir, load_dir, 'y_train.npy')
    y_val_file = os.path.join(root_dir, load_dir, 'y_val.npy')
    if os.path.isfile(x_train_file_1) and os.path.isfile(x_train_file_2) and os.path.isfile(y_train_file) and \
            os.path.isfile(x_val_file_1) and os.path.isfile(x_val_file_2) and os.path.isfile(y_val_file):
        x_train_1 = np.load(x_train_file_1)
        x_train_2 = np.load(x_train_file_2)
        x_val_1 = np.load(x_val_file_1)
        x_val_2 = np.load(x_val_file_2)
        y_train = np.load(y_train_file)
        y_val = np.load(y_val_file)
        print("Finish data loading")
    else:
        # Load Data
        x_train_1 = []
        x_train_2 = []
        x_val_1 = []
        x_val_2 = []
        y_train = []
        y_val = []
        mean_dict_1 = {}
        std_dict_1 = {}
        mean_dict_2 = {}
        std_dict_2 = {}
        for stock_id in stock_ids:
            if type(stock_id) in [tuple, list]:
                stock_id = stock_id[0]
            # 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量
            select = "SELECT 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量, \
                      外資3, 外資5, 外資10, 外資20, 外資60, 法人3, 法人5, 法人10, 法人20, 法人60, 融資3, 融資5, 融資10, 融資20, 融資60 \
                      FROM StockAll \
                      WHERE 代號 = '" + str(stock_id) + "' \
                      AND 日期 <= '2019-12-31' \
                      AND 日期 >= '2006-01-01' \
                      ORDER BY 日期 DESC \
                      LIMIT 4000 \
                      ;"
            values = list(stock_sql.read_stock_values(select))
            # 移除 MA20, K9, D9, OSC, UB20, PB, BW 為NULL的資料
            values = [v for v in values if 0 not in v[6:-1] and None not in v[6:-1]]
            if len(values) < 2000:
                print('(False) Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
                continue

            # Create X Input 1
            x_data_1 = np.array(values)[:, 1:13].astype(np.float)
            mean_1 = x_data_1.mean(axis=0)
            std_1 = x_data_1.std(axis=0)
            x_data_1 = (x_data_1[30:] - mean_1) / std_1
            mean_dict_1[stock_id] = mean_1
            std_dict_1[stock_id] = std_1
            # Create X Input 2
            x_data_2 = np.array(values)[:, 13:].astype(np.float)
            mean_2 = x_data_2.mean(axis=0)
            std_2 = x_data_2.std(axis=0)
            x_data_2 = (x_data_2[30:] - mean_2) / std_2
            mean_dict_2[stock_id] = mean_2
            std_dict_2[stock_id] = std_2

            # Create Y data
            y_data = []
            for idx in range(30, len(values), 1):
                if values[idx][4] * 1.2 < values[idx - 30][4]:
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
                y_train += y_data[split_point:]
            x_val_1.append(x_data_1[:split_point])
            x_val_2.append(x_data_2[:split_point])
            y_val += y_data[:split_point]
            print('Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))

        # Combine all stocks
        x_train_1 = np.concatenate(x_train_1)
        x_train_2 = np.concatenate(x_train_2)
        x_val_1 = np.concatenate(x_val_1)
        x_val_2 = np.concatenate(x_val_2)

        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # Save data at local
        if not os.path.isdir(os.path.join(root_dir, load_dir)):
            os.makedirs(os.path.join(root_dir, load_dir))
        np.save(os.path.join(root_dir, load_dir, 'mean_1.npy'), mean_dict_1)
        np.save(os.path.join(root_dir, load_dir, 'mean_2.npy'), mean_dict_2)
        np.save(os.path.join(root_dir, load_dir, 'std_1.npy'), std_dict_1)
        np.save(os.path.join(root_dir, load_dir, 'std_2.npy'), std_dict_2)
        np.save(os.path.join(root_dir, load_dir, 'x_train_1.npy'), x_train_1)
        np.save(os.path.join(root_dir, load_dir, 'x_train_2.npy'), x_train_2)
        np.save(os.path.join(root_dir, load_dir, 'x_val_1.npy'), x_val_1)
        np.save(os.path.join(root_dir, load_dir, 'x_val_2.npy'), x_val_2)
        np.save(os.path.join(root_dir, load_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(root_dir, load_dir, 'y_val.npy'), y_val)
        print("Finish data loading")
    x_train = (x_train_1, x_train_2)
    x_val = (x_val_1, x_val_2)
    return x_train, y_train, x_val, y_val


def load_and_pred_data(stock_sql, model):
    ret = []
    root_dir = os.path.join('dataset', os.path.basename(__file__).split('.')[0])
    version = int(os.path.basename(__file__).split('.')[0].split('_')[-1])

    mean_dict_1 = np.load(os.path.join(root_dir, '1723', 'mean_1.npy'), allow_pickle=True).item()
    mean_dict_2 = np.load(os.path.join(root_dir, '1723', 'mean_2.npy'), allow_pickle=True).item()
    std_dict_1 = np.load(os.path.join(root_dir, '1723', 'std_1.npy'), allow_pickle=True).item()
    std_dict_2 = np.load(os.path.join(root_dir, '1723', 'std_2.npy'), allow_pickle=True).item()
    # Run all dates
    for stock_id in mean_dict_1.keys():
        select = "SELECT 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量, \
                  外資3, 外資5, 外資10, 外資20, 外資60, 法人3, 法人5, 法人10, 法人20, 法人60, 融資3, 融資5, 融資10, 融資20, 融資60 \
                  FROM StockAll \
                  WHERE 代號 = '" + str(stock_id) + "' \
                  AND 日期 >= '2006-01-01' \
                  ORDER BY 日期 DESC \
                  LIMIT 4000 \
                  ;"

        values = list(stock_sql.read_stock_values(select))
        # 移除 MA20, K9, D9, OSC, UB20, PB, BW, 外資... 為NULL的資料
        values = [v for v in values if 0 not in v[6:-1] and None not in v[6:-1]]
        if len(values) < 1:
            print('(False) Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
            continue

        # Create X Input 1
        x_data_1 = np.array(values)[:, 1:13].astype(np.float)
        x_data_1 = (x_data_1 - mean_dict_1[stock_id]) / std_dict_1[stock_id]

        # Create X Input 2
        x_data_2 = np.array(values)[:, 13:].astype(np.float)
        x_data_2 = (x_data_2 - mean_dict_2[stock_id]) / std_dict_2[stock_id]

        # Predict
        outputs = model.predict((x_data_1, x_data_2))

        # Save predict result
        dates = [str(v[0]) for v in values]
        for date, output in zip(dates, outputs):
            ret.append((version, date, stock_id, "{:.4f}".format(output[0])))
        print('Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
    return ret


if __name__ == '__main__':
    from units.sql import StockSQL

    # 連線SQL資料庫
    stock_sql_test = StockSQL()

    # 載入資料
    load_data(stock_sql_test, load_dir='171')
