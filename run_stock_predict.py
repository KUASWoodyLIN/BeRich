import os
from data import *
from units.model import *
from units.sql import StockSQL


FILE = '2021-1-18-1723-version6.h5'
WEIGHT_PATH = os.path.join('output', FILE)
VERSION = FILE.split('version')[-1].split('.h5')[0]


def main():
    stock_sql = StockSQL()
    if VERSION == 1:
        model = load_dnn_model_v1(weight_path=WEIGHT_PATH)
        ret = load_and_pred_data_v1(stock_sql, model)
    elif VERSION == 2:
        model = load_dnn_model_v2(weight_path=WEIGHT_PATH)
        ret = load_and_pred_data_v2(stock_sql, model)
    elif VERSION == 3:
        model = load_dnn_model_v3(weight_path=WEIGHT_PATH)
        ret = load_and_pred_data_v3(stock_sql, model)
    elif VERSION == 4:
        model = load_dnn_model_v4(weight_path=WEIGHT_PATH)
        ret = load_and_pred_data_v4(stock_sql, model)
    elif VERSION == 5:
        model = load_dnn_model_v5(weight_path=WEIGHT_PATH)
        ret = load_and_pred_data_v5(stock_sql, model)
    else:
        model = load_dnn_model_v6(weight_path=WEIGHT_PATH)
        ret = load_and_pred_data_v6(stock_sql, model)

    print("上傳資料中.....")
    # 上傳預測結果
    stock_sql.write_multi_line(
        "INSERT INTO `AI__分析a` VALUES (%s,%s,%s,%s,%s)",
        ret
    )
    print("Finish")


if __name__ == "__main__":
    main()
