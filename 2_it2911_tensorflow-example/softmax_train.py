import tensorflow as tf

import os

# 变量权值，矩阵，每个特征权值列对应一个输出类别
w = tf.Variable(tf.zeros([4, 3]), name="weights")
# 模型偏置，每个偏置对应一个输出类别
b = tf.Variable(tf.zeros([3]), name="bias")

# 输入值合并
def combine_inputs(X):
    print("function: combine_inputs")
    return tf.matmul(X, w) + b

# 计算返回推断模型输出(数据X)
def inference(X):
    print("function: inference")
    # 调用softmax分类函数
    return tf.nn.softmax(combine_inputs(X))

# 计算损失(训练数据X及期望输出Y)
def loss(X, Y):
    print("function: loss")
    # 使用最简单的最小均方误差（MSE：minimum squared error）作为成本函数。
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

# 从csv文件读取数据，加载解析，创建批次读取张量多行数据
def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    # 字符串(文本行)转换到指定默认值张量列元组，为每列设置数据类型
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # 读取文件，加载张量batch_size行
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

# 读取或生成训练数据X及期望输出Y
def inputs():
    print("function: inputs")

    # 数据来源：UCI Machine Learning Repository: Iris Data Set
    # iris.data改为iris.csv，增加sepal_length, sepal_width, petal_length, petal_width, label字段行首行
    sepal_length, sepal_width, petal_length, petal_width, label = read_csv(100, "iris.csv", [[0.0], [0.0], [0.0], [0.0], [""]])

    # 转换属性数据 将类名称转抽象为从0开始的类别索引
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, ["Iris-virginica"])
    ])), 0))

    # 特征装入矩阵，转置，每行一样本，每列一特征
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

    return features,

#训练或调整模型参数(计算总损失)
def train(totle_loss):
    print("function: train")
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(totle_loss)

def evaluate(sess, X, Y):
    print("function: train")
    # 选择预测输出值最大概率类别
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    # 统计所有正确预测样本数，除以批次样本总数，得到正确预测百分比
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

#会话对象启动数据流图，搭建流程
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

    for step in range(training_steps):
        # 实际训练
        sess.run([train_op])
        # 查看训练过程损失递减
        if step % 10 == 0:
            print(str(step)+ " loss: ", sess.run([total_loss]))

    print(str(training_steps) + " final loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()