import tensorflow as tf#导入TensorFlow库

# 构建数据流图
# 显式创建Graph对象
graph = tf.Graph()
# 设为默认Graph对象
with graph.as_default():

    # 创建Variable对象名称作用域
    with tf.name_scope("variables"):
        # 记录数据流图运行次数的Variable对象，初值为0，数据类型为32位整型，不可自动修改，以global_step标识
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        # 追踪模型所有输出累加和的Variable对象，初值为0.0，数据类型为32位浮点型，不可自动修改，以total_output标识
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

    # 创建变换计算Op名称作用域
    with tf.name_scope("transformation"):
        # 创建独立输入层名称作用域
        with tf.name_scope("input"):
            # 创建占位符，接收一个32位浮点型任意长度的向量作为输入，以input_placeholder_a标识
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")

        # 创建独立中间层名称作用域
        with tf.name_scope("intermediate_layer"):
            # 创建创建归约乘积Op，接收张量输入，输出张量所有分量(元素)的乘积，以product_b标识
            b = tf.reduce_prod(a, name="product_b")
            # 创建创建归约求和Op，接收张量输入，输出张量所有分量(元素)的求和，以sum_c标识
            # http://blog.csdn.net/lenbow/article/details/52152766
            c = tf.reduce_sum(a, name="sum_c")
        # 创建独立输出层名称作用域
        with tf.name_scope("output"):
            # 创建创建求和Op，接收两个标量输入,输出标量求和,以output标识
            output = tf.add(b, c, name="output")

    with tf.name_scope("update"):
        # 用最新的输出更新Variable对象total_output
        update_total = total_output.assign_add(output)
        # 增1更新Variable对象global_step，记录数据流图运行次数
        increment_step = global_step.assign_add(1)

    # 创建数据汇总Op名称作用域
    with tf.name_scope("summaries"):
        # 计算平均值，输出累加和除以数据流图运行次数，把运行次数数据类型转换为32位浮点型，以average标识
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
        # 创建输出节点标量数据统计汇总，以output_summary标识
        tf.summary.scalar('output_summary',output)
        # 创建输出累加求和标量数据统计汇总，以total_summary标识
        tf.summary.scalar('total_summary',update_total)
        # 创建平均值标量数据统计汇总，以average_summary标识
        tf.summary.scalar('average_summary',avg)

    # 创建全局Operation(Op)名称作用域
    with tf.name_scope("global_ops"):
        # 创建初始化所有Variable对象的Op
        init = tf.global_variables_initializer()
        # 创建合并所有汇总数据的Op
        merged_summaries = tf.summary.merge_all()

    #运行数据流图
    # 用显式创建Graph对象启动Session会话对象
    sess = tf.Session(graph=graph)
    # 启动FileWriter对象，保存汇总数据
    writer = tf.summary.FileWriter('./improved_graph', graph)
    # 运行Variable对象初始化Op
    sess.run(init)


    # 定义数据注图运行辅助函数
    def run_graph(input_tensor):

        """
        辅助函数：用给定的输入张量运行数据流图，
        并保存汇总数据
        """
        # 创建feed_dict参数字典，以input_tensor替换a句柄的tf.placeholder节点值
        feed_dict = {a: input_tensor}
        # 使用feed_dict运行output不关心存储，运行increment_step保存到step，运行merged_summaries Op保存到summary
        _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
        # 添加汇总数据到FileWriter对象，global_step参数时间图示折线图横轴
        writer.add_summary(summary, global_step=step)

    #用不同的输入用例运行数据流图

    run_graph([2,8])

    run_graph([3,1,3,3])

    run_graph([8])

    run_graph([1,2,3])

    run_graph([11,4])

    run_graph([4,1])

    run_graph([7,3,1])

    run_graph([6,3])

    run_graph([0,2])

    run_graph([4,5,6])
    # 将汇总数据写入磁盘
    writer.flush()
    # 关闭FileWriter对象，释放资源
    writer.close()
    # 关闭Session对象，释放资源
    sess.close()
