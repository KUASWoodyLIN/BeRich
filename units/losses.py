import tensorflow as tf


# def binary_focal_loss(y_true, y_pred, alpha=0.9, gamma_true=2.0, gamma_false=2.0):
#     alpha_factor = tf.ones_like(y_true, dtype=tf.float32) * alpha
#     alpha_factor = tf.where(tf.equal(y_true, 1), alpha_factor, 1-alpha_factor)
#     gamma_t = tf.ones_like(y_true, dtype=tf.float32) * gamma_true
#     gamma_f = tf.ones_like(y_true, dtype=tf.float32) * gamma_false
#     gamma_factor = tf.where(tf.equal(y_true, 1), gamma_t, gamma_f)
#     focal_weight = tf.where(tf.equal(y_true, 1), 1 - y_pred, y_pred)
#     focal_weight = alpha_factor * focal_weight ** gamma_factor
#     losses = focal_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)
#     # binary cross entropy
#     # x = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
#     print(losses)
#     return losses


def binary_focal_loss(y_true, y_pred, alpha_true=10, alpha_false=1, gamma_true=2.0, gamma_false=2.0):
    alpha_true = tf.ones_like(y_true, dtype=tf.float32) * alpha_true
    alpha_false = tf.ones_like(y_true, dtype=tf.float32) * alpha_false
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha_true, alpha_false)
    gamma_t = tf.ones_like(y_true, dtype=tf.float32) * gamma_true
    gamma_f = tf.ones_like(y_true, dtype=tf.float32) * gamma_false
    gamma_factor = tf.where(tf.equal(y_true, 1), gamma_t, gamma_f)
    focal_weight = tf.where(tf.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma_factor
    losses = focal_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)
    # binary cross entropy
    # x = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    return losses
