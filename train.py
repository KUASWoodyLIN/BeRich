import os
import datetime
import argparse
from data import *
from units.model import *
from units.sql import StockSQL
from units.metrics import OverKill, UnderKill
from units.losses import binary_focal_loss
from units.callbacks import SaveBastOverKillAddUnderKillModel, OverKillAddUnderKill


parse = argparse.ArgumentParser()
parse.add_argument('--version', dest='version', type=int, default=7)
parse.add_argument('--data-mode', dest='data_mode', type=str, default='1723')
args = parse.parse_args()

VERSION = args.version                 # 1, 2 or 3
TRAINING_DATA = args.data_mode       # '171' or '1723'
TEST_MODE = False


def main():
    # 初始話參數
    if TEST_MODE:
        today = 'test'
    else:
        dt = datetime.datetime.today()
        today = "{}-{}-{}".format(dt.year, dt.month, dt.day)
    model_dir = os.path.join('outputs', '{}-{}-version{}.h5'.format(today, TRAINING_DATA, VERSION))
    log_dir = os.path.join('logs', '{}-{}-version{}'.format(today, TRAINING_DATA, VERSION))
    if os.path.isdir(log_dir) or os.path.isfile(model_dir):
        print("檔案已經存在")
        exit()

    # 連線SQL資料庫
    stock_sql = StockSQL()

    # 載入資料 & 創建Model
    if VERSION == 1:
        x_train, y_train, x_val, y_val = load_data_v1(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v1()
    elif VERSION == 2:
        x_train, y_train, x_val, y_val = load_data_v2(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v2()
    elif VERSION == 3:
        x_train, y_train, x_val, y_val = load_data_v3(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v3()
    elif VERSION == 4:
        x_train, y_train, x_val, y_val = load_data_v4(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v4()
    elif VERSION == 5:
        x_train, y_train, x_val, y_val = load_data_v5(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v5()
    elif VERSION == 6:
        x_train, y_train, x_val, y_val = load_data_v6(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v6()
    else:
        x_train, y_train, x_val, y_val = load_data_v7(stock_sql, load_dir=TRAINING_DATA)
        model = load_dnn_model_v7()

    # Training
    model.compile(
        tf.keras.optimizers.Adam(0.001),
        loss=binary_focal_loss,
        metrics=[
            keras.metrics.BinaryAccuracy(),
            OverKill(),
            UnderKill()
        ],
        run_eagerly=True
    )

    model_tcbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = SaveBastOverKillAddUnderKillModel(model_dir)
    model_acbk = OverKillAddUnderKill(log_dir)
    model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[model_tcbk, model_mckp, model_acbk]
    )


if __name__ == "__main__":
    main()
