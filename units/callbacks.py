import os
import numpy as np
import tensorflow as tf


class SaveBastOverKillAddUnderKillModel(tf.keras.callbacks.Callback):
    def __init__(self, weights_file, mode='min', save_weights_only=False):
        super(SaveBastOverKillAddUnderKillModel, self).__init__()
        self.weights_file = weights_file
        self.mode = mode
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.best = np.Inf
        else:
            self.best = -np.Inf

    def save_model(self):
        if self.save_weights_only:
            self.model.save_weights(self.weights_file)
        else:
            self.model.save(self.weights_file)
        print("Save", self.weights_file)

    def on_epoch_end(self, epoch, logs=None):
        overkill = logs.get('val_over_kill')
        underkill = logs.get('val_under_kill')
        monitor_value = overkill + underkill
        if self.mode == 'min' and monitor_value < self.best:
            self.save_model()
            self.best = monitor_value
        elif self.mode == 'max' and monitor_value > self.best:
            self.save_model()
            self.best = monitor_value


class OverKillAddUnderKill(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(OverKillAddUnderKill, self).__init__()
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        # path = os.path.join(self.log_dir, 'confusion_matrix')
        # # 創建TensorBoard紀錄檔
        # self.writer = tf.summary.create_file_writer(path)

        path_train = os.path.join(self.log_dir, 'train')
        path_val = os.path.join(self.log_dir, 'validation')
        # 創建TensorBoard紀錄檔
        self.writer_train = tf.summary.create_file_writer(path_train)
        self.writer_val = tf.summary.create_file_writer(path_val)


    def on_epoch_end(self, epoch, logs=None):
        train_overkill = logs.get('over_kill')
        train_underkill = logs.get('under_kill')
        val_overkill = logs.get('val_over_kill')
        val_underkill = logs.get('val_under_kill')
        # # 將圖片紀錄在TensorBoard log中
        # with self.writer.as_default():
        #     tf.summary.scalar('OverKill Add UnderKill/train', train_overkill + train_underkill, step=epoch)
        #     tf.summary.scalar('OverKill Add UnderKill/validation', val_overkill + val_underkill, step=epoch)
        # 將圖片紀錄在TensorBoard log中
        with self.writer_train.as_default():
            tf.summary.scalar('OverKill Add UnderKill/train', train_overkill + train_underkill, step=epoch)
        with self.writer_val.as_default():
            tf.summary.scalar('OverKill Add UnderKill/validation', val_overkill + val_underkill, step=epoch)
