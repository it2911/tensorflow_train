import tensorflow as tf

import os

# y = wx + b
# 一元一次方程为基础的回归条件式，转化为tf工程

# w变量
w = tf.Variable(tf.zeros([2, 1], dtype="float32", name="weights"))
# b变量
b = tf.Variable(0, dtype="float32", name="bias")

def inference(X): # module function
    print("计算返回推断模型输出inference")
    return tf.matmul(X, w) + b

def loss(X, Y): # loss function
    print("计算损失，损失标准，方差之和")
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs(): # data input funciton
    print("输入数据")
    #Data from http: // people.sc.fsu.edu / ~jburkardt / datasets / regression / x09.txt

    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]

    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308,
                         220, 311, 181, 274, 303, 244]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)

def train(total_lost): # train function
    print("训练或调整模型参数(计算总损失)")
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_lost)

def evaluate(sess, X, Y):# evaluate module function
    print("评估训练模型")
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))

# 训练模型，多个训练周期更新参数(变量)。
# tf.train.Saver类保存变量到二进制文件。周期性保存所有变量，创建检查点(checkpoint)文件，从最近检查点恢复训练。
# 启动会话对象前实例化Saver对象，每完成1000次训练迭代或训练结束，调用tf.train.Saver.save方法，
# 创建遵循命名模板my-model{step}检查点文件，保存每个变量当前值，默认只保留最近5个文件，更早的自动删除。
saver = tf.train.Saver()

with tf.Session() as sess:
    print("开始制作流程图")
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    # tf.train.start_queue_runners 这个函数将会启动输入管道的线程，填充样本到队列中，以便出队操作可以从队列中拿到样本。
    # 这种情况下最好配合使用一个tf.train.Coordinator，这样可以在发生错误的情况下正确地关闭这些线程。
    # 协调器和队列运行器(Coordinator and QueueRunner)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    training_steps = 1000 # 实际迭代次数
    initial_step = 0

    # tf.train.get_checkpoint_state函数会通过checkpoint文件自动
    # 找到目录中最新模型的文件名。
    #checkpoint_dir = "./"
    #ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir))

    # tf. train.get_checkpoint_state方法验证有无检查点文件。

    #    print("Checkpoint path:" + ckpt.model_checkpoint_path)
        # tf. trainSaver.restore方法恢复变量值。
        # 从检查点恢复模型参数
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    for step in range(initial_step, training_steps):  # 实际训练闭环
        sess.run([train_op])

        # 查看训练过程损失递减
        if step % 10 == 0:
            print(str(step) + "loss: ", sess.run([total_loss]))
            #save_file = saver.save(sess, "my-model", global_step=step)
            #print(str(step) + "save_file: ", save_file)

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    #saver.save(sess, "my-model", global_step=training_steps)
    print(str(training_steps) + "final loss: ", sess.run([total_loss]))
    sess.close()