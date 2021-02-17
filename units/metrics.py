import tensorflow as tf


class OverKill(tf.keras.metrics.Metric):
    def __init__(self, name='over_kill', **kwargs):
        super(OverKill, self).__init__(name=name, **kwargs)
        self.tn = self.add_weight('True Positive', initializer='zeros')
        self.fp = self.add_weight('False Positive ', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tn = tf.reduce_sum(tf.where((y_pred < 0.5) & (y_true == 0), 1., 0.))
        fp = tf.reduce_sum(tf.where((y_pred > 0.5) & (y_true == 0), 1., 0.))
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)

    def result(self):
        # 計算準確率
        return tf.math.divide_no_nan(self.fp, self.tn + self.fp)

    def reset_states(self):
        # 每一次Epoch結束後會重新初始化變數
        self.tn.assign(0.)
        self.fp.assign(0.)


class UnderKill(tf.keras.metrics.Metric):
    def __init__(self, name='under_kill', **kwargs):
        super(UnderKill, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight('True Negative', initializer='zeros')
        self.fn = self.add_weight('False Negative', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = tf.reduce_sum(tf.where((y_pred > 0.5) & (y_true == 1), 1., 0.))
        fn = tf.reduce_sum(tf.where((y_pred < 0.5) & (y_true == 1), 1., 0.))
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        # 計算準確率
        return tf.math.divide_no_nan(self.fn, self.tp + self.fn)

    def reset_states(self):
        # 每一次Epoch結束後會重新初始化變數
        self.tp.assign(0.)
        self.fn.assign(0.)
