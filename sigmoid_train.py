import tensorflow as tf

import os
import time

# 变量权值
w = tf.zeros([1, 5], name="weights")
# 线性函数常量，模型偏置
b = tf.Variable(0, name="bias")

# 输入值合并
def combine_inputs(X):
    print("function: combine_inputs")
    return tf.matmul(X, w) + b

# 计算返回推断模型输出(数据X)
def inference(X):
    print("function: inference")
    return tf.sigmoid(combine_inputs(X))

# 计算损失(训练数据X及期望输出Y)
# 交叉熵(cross entropy)
def loss(X, Y):
    print("function: loss")
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

#从csv文件读取数据，加载解析，创建批次读取张量多行数据
def read_csv(batch_size, file_name, record_defaults):
    print("function: read_csv")
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decode = tf.decode_csv(value, record_defaults=record_defaults)
    # 读取文件，加载张量batch_size行
    return tf.train .shuffle_batch(decode, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

def inputs():
    print("function: inputs")
    # 数据来源：https://www.kaggle.com/c/titanic/data
    # 模型依据乘客年龄、性别、船票等级推断是否能够幸存
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    # 转换属性数据
    # 一等票
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    # 二等票
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    # 三等票
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    # 性别，男性为0，女性为1
    gender = tf.to_float(tf.equal(sex, ["female"]))
    #所有特征排列矩阵，矩阵转置，每行一样本，每列一特征
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    return features, survived

# 训练或调整模型参数(计算总损失)
def train(total_loss):
    print("function: train")
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    print("function: evaluate")
    # 样本输出大于0.5转换为正回答
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    # 统计所有正确预测样本数，除以批次样本总数，得到正确预测百分比
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# 会话对象启动数据流图，搭建流程
with tf.Session() as sess:
    print("Session: start")
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    # 实际训练迭代次数
    training_steps = 1000
    # 实际训练闭环
    for step in range(training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print(str(step) + " loss: ", sess.run([total_loss]))

    print(str(training_steps) + " final loss: ", sess.run([total_loss]))
    evaluate(sess, X, Y)
    time.sleep(5)
    coord.request_stop()
    coord.join(threads)
    sess.close()