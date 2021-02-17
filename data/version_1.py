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
    x_train_file = os.path.join(root_dir, load_dir, 'x_train.npy')
    y_train_file = os.path.join(root_dir, load_dir, 'y_train.npy')
    x_val_file = os.path.join(root_dir, load_dir, 'x_val.npy')
    y_val_file = os.path.join(root_dir, load_dir, 'y_val.npy')
    if os.path.isfile(x_train_file) and os.path.isfile(y_train_file) and os.path.isfile(x_val_file) and os.path.isfile(y_val_file):
        x_train = np.load(x_train_file)
        y_train = np.load(y_train_file)
        x_val = np.load(x_val_file)
        y_val = np.load(y_val_file)
        print("Finish data loading")
    else:
        # Load Data
        x_train = []
        x_val = []
        y_train = []
        y_val = []
        mean_dict = {}
        std_dict = {}
        for stock_id in stock_ids:
            if type(stock_id) in [tuple, list]:
                stock_id = stock_id[0]
            # 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量
            select = "SELECT 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量 \
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
            if len(values) < 2000 or stock_id in [6222, 8121]:
                print('(False) Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
                continue

            # Create X data
            x_data = np.array(values)[:, 1:].astype(np.float)
            mean = x_data.mean(axis=0)
            std = x_data.std(axis=0)
            x_data = (x_data[30:] - mean) / std

            mean_dict[stock_id] = mean
            std_dict[stock_id] = std

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
                x_train.append(x_data[split_point:])
                y_train += y_data[split_point:]
            x_val.append(x_data[:split_point])
            y_val += y_data[:split_point]
            print('Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))

        # Combine all stocks
        x_train = np.concatenate(x_train)
        x_val = np.concatenate(x_val)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # Save data at local
        if not os.path.isdir(os.path.join(root_dir, load_dir)):
            os.mkdir(os.path.join(root_dir, load_dir))
        np.save(os.path.join(root_dir, load_dir, 'mean.npy'), mean_dict)
        np.save(os.path.join(root_dir, load_dir, 'std.npy'), std_dict)
        np.save(os.path.join(root_dir, load_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(root_dir, load_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(root_dir, load_dir, 'x_val.npy'), x_val)
        np.save(os.path.join(root_dir, load_dir, 'y_val.npy'), y_val)
        print("Finish data loading")
    return x_train, y_train, x_val, y_val


def load_and_pred_data(stock_sql, model):
    ret = []
    root_dir = os.path.join('dataset', os.path.basename(__file__).split('.')[0])
    version = int(os.path.basename(__file__).split('.')[0].split('_')[-1])

    mean_dict = np.load(os.path.join(root_dir, '1723', 'mean.npy'), allow_pickle=True).item()
    std_dict = np.load(os.path.join(root_dir, '1723', 'std.npy'), allow_pickle=True).item()
    # Run all dates
    for stock_id in mean_dict.keys():
        select = "SELECT 日期, 開, 高, 低, 收, MA20, K9, D9, OSC, UB20, PB, BW, 成交量 \
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
        if len(values) < 1:
            print('(False) Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
            continue

        # Create X Input 1
        x_data = np.array(values)[:, 1:].astype(np.float)
        x_data = (x_data - mean_dict[stock_id]) / std_dict[stock_id]

        # Predict
        outputs = model.predict(x_data)

        # Save predict result
        dates = [str(v[0]) for v in values]
        for date, output in zip(dates, outputs):
            ret.append((version, date, stock_id, "{:.4f}".format(output[0])))
        print('Preparing Stock: {}, Number: {}'.format(stock_id, len(values)))
    return ret
