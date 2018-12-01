import tensorflow as tf



def gauss_of_lin(x):
    return tf.exp(-1*(tf.abs(x)))

def gauss_times_linear(x):
    return tf.exp(-6.*tf.sqrt(tf.abs(x)+1e-4))*x*3.*36 

def sinc(x):
    return tf.where(tf.equal(x, tf.zeros_like(x)), tf.zeros_like(x) + 1., tf.sin(x)/(x))

def open_tanh(x):
    return 0.9*tf.nn.tanh(x)+0.1*x


