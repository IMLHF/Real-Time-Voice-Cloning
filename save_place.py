import tensorflow as tf

config = tf.ConfigProto()
# config.allow_soft_placement = True
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction = 0.43
# w = tf.get_variable("w",[10000,100000])
init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)
while True:
    input()
# w = sess.run(w)
# print(w)
