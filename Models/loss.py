#The Smooth L1 loss

from keras import backend as K
import tensorflow as tf

HUBER_DELTA=0.5
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)
