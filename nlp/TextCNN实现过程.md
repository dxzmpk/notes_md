# TextCNN实现

## 和传统模型的对比

### n-grams

当词典比较大时，计算n大于3时代价迅速上升（?）

### CNN

卷积神经网络可以自动学得单词的很好的表示，而不需要表示整个词典

而且，当卷积核的尺寸大于5时，可以得到类似于5-gram的结果，而且不止限于此，可以包含更多的信息，但是和传统方法比将表示压缩了。

## CNN超参数的选择

### 词的表示

- word2vec
- GloVe
- one-hot

### 是否使用zero-padding

添加边缘补零也叫做wide convolution, 否则就是narrow convolution

边缘补零后，输入和输出之间的计算公式为：
$$
n_{out} = (n_{in} + 2*n_{padding}-(n_{filter}-1))
$$

### 步长

决定每一步移动核矩阵几格，步长越大，核函数应用得越少，得到的输出尺寸也越小。一般使用的步长为1，但是步长增大可能会使cnn具有和rnn相似的效果

### 核函数的数量和尺寸

### 池化层

最常见的池化层是对每一个核函数的结果都进行最大池化。池化层也可以形如2*2的窗口，每一个窗口产生一个值。

池化层的一个作用是返回固定长度的矩阵，通常在分类前需要这个操作。不管输入的尺寸有多少，以及每个核函数的维数是多大，最终产生的维数都由核函数的个数决定，而这在池化层完成。另一个作用是减少矩阵的维数，但同时保存最突出的信息。在自然语言处理中，当你想测定句子中是否有某个词组时，如果词组在句子中出现，核函数经过那个词组是会产生一个很大的值，执行最大池化后，这个词组存在的信息被保留下来，但是它在哪里出现的信息被丢失了。

### 激活函数

- ReLU
- tanh

### 通道

通道是输入数据的不同“视图”。例如，在图像识别中R\G\B通道，你可以在多通道进行卷积操作，核函数可以相同也可以不同。对于自然语言处理任务，也可以使用多通道，例如使用单词的不同表示（word2vec和GloVe）作为不同的通道

## 适用领域

### 分类任务

情感分析，垃圾邮件检测(spam detection)，话题标注。

由于卷积核池化操作会丢失单词的顺序信息，所以序列标注问题(PoS Tagging)或者命名实体识别很难被纯的CNN架构实现。（可以通过在输入中添加位置信息来实现)

[关系抽取和关系分类](https://cs.nyu.edu/~thien/pubs/vector15.pdf)，作者使用单词相对于感兴趣实体的相对位置作为卷积层的输入。

[找出对于信息抽取有用的语义信息表示](https://www.microsoft.com/en-us/research/publication/a-latent-semantic-model-with-convolutional-pooling-structure-for-information-retrieval/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F226585%2Fcikm2014_cdssm_final.pdf) [论文前置信息](http://research.microsoft.com/pubs/226584/604_Paper.pdf)，例如基于读者正在阅读的材料推荐其有可能感兴趣的文件。

[通过CNN训练词向量](http://emnlp2014.org/papers/pdf/EMNLP2014194.pdf)

字符级别的CNN   

- Speech tagging [链接](http://jmlr.org/proceedings/papers/v32/santos14.pdf)  
- [Sentiment Analysis](http://arxiv.org/abs/1509.01626) and [Text Categorization](http://arxiv.org/abs/1509.01626)  

- [Language Modeling +LSTM](http://arxiv.org/abs/1508.06615)

![Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](http://www.wildml.com/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-8.03.47-AM-1024x413.png) 

## 其他Tips

1. 最大池化的效果总是由于平均池化

2. 在nlp任务中，正则化没什么用
3. filter尺寸很关键，不同任务之间最理想的尺寸不同。
4. 字符级别的cnn在大数据集上工作的好，但是在简单模型的简单数据上表现不佳。

## 模型的实现 [论文](https://arxiv.org/abs/1408.5882)

### 1.处理数据

1. 加载正例和反例
2. 清洗数据
3. 将每个评价的长度用`<pad>`标签增加为最大评价的长度，因为同一个batch中的数据长度需要一致，这样才能同步运行加快速度。
4. 建立单词索引，将单词映射为18765的整数，这样每个句子就是整数类型的`59*1维`向量。

### 2.建立TextCNN类，存储超参数

- sequence_length : 句子的长度，这里是59
- num_classes ： 输出的种类数，2
- vocab_size ： 词典的尺寸，embedding layer的宽度， 其形状是`[vocabulary_size,embedding_size]`
- embedding_size
- filter_sizes , 核矩阵的尺寸，[3,4,5]代表有三种长度的核函数
- num_filters：每一种核函数的数量

```python
import tensorflow as tf
import numpy as np

class TextCNN(object):
"""
A CNN for text classification.
Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
"""
def __init__(
self, sequence_length, num_classes, vocab_size,
embedding_size, filter_sizes, num_filters):
# Implementation…
```



### 3. 输入占位符: input placeholders

```python
# Placeholders for input, output and dropout
# 输入的占位符不仅要为train占位，还有test, 还有待预测的数据.None means anything,batch_size
self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
```

### 4. 词向量表示层(embedding layer)

```python
with tf.device(‘/cpu:0’), tf.name_scope("embedding"):
    W = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
    name="W")
    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
```

其中，` tf.name_scope `创建了一个新的域，这个域将所有的操作都集中在该节点下，便于在TensorBoard中可视化神经网络。W是最终学得的embedding矩阵，通过random_uniform函数初始化，tf.nn.embedding_lookup创建embedding操作，结果是一个形状为`[None,sequence_length,embedding_size]`的`Tensor`, TF的conv2d操作需要参数为4维的`Tensor`，对应`[batch,width,height,channel]`,embedding的结果不包含channel维，所以在最后一维扩展一下。

### 5.词向量表示层

```python
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
	with tf.name_scope("conv-maxpool-%s" % filter_size):
	# Convolution Layer
		filter_shape = [filter_size, embedding_size, 1, num_filters]
		W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), 					name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
        self.embedded_chars_expanded,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
        # Apply nonlinearity h是对卷积层的输出非线性化之后的结果
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
        h,
        ksize=[1, sequence_length – filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding=’VALID’,#不使用边缘扩展padding技术
        name="pool")
        # pooled 是维度为[batch_size, 1, 1, num_filters]的张量
    	pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
self.h_pool = tf.concat(3, pooled_outputs)
self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
# -1代表尽可能地铺开维度（?）
```

### 6. DropOut层

```python
# Add dropout
with tf.name_scope("dropout"):
	self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
```

目的是防止神经元之间相互协作，而强迫其单独学习有用的特征，` dropout_keep_prob `规定保留率。这里在训练时调成0.5,而在测试时调成1.0

### 7.打分和预测

```python
with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
    self.predictions = tf.argmax(self.scores, 1, name="predictions")
```

  `tf.nn.xw_plus_b` 时执行 $Wx+b$矩阵乘法的缩写。

### 8.损失和正确率

```python
# Calculate mean cross-entropy loss
with tf.name_scope("loss"):
losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)#这里的score时二维的，如何计算误差？
self.loss = tf.reduce_mean(losses)
```

### 9.训练模型 - 创建默认Session和默认图

```python
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
    # Code that operates on the default graph and session comes here…
```

 `allow_soft_placement`  允许当硬件不满足条件时自动调整，否则会产生error。` log_device_placement ` 记录操作在哪个硬件上完成。这里FLAGS来自命令行参数。

### 10.开始训练 - 新建模型

```python
cnn = TextCNN(
    sequence_length=x_train.shape[1],
    num_classes=2,
    vocab_size=len(vocabulary),
    embedding_size=FLAGS.embedding_dim,
    filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
    num_filters=FLAGS.num_filters)
```

定义优化损失损失函数的方法(调参数):Adam方法

```python
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-4)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
```

每次执行train_op都会进行梯度下降法调参，TF可以自动得知哪个参数时可训练的并计算其梯度， `global_step`  用来计数训练的步骤，即执行train_op的次数。

### 11.训练模型 - 过程统计

Summaries可以帮助我们跟踪在训练和验证过程中的一些数据。例如，我们可以跟踪accuracy和error的变化。还可以跟踪`  histograms of layer activations `等。 Summaries是序列化对象，通过`SummaryWriter`写到磁盘上。

```python
# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# Summaries for loss and accuracy
loss_summary = tf.scalar_summary("loss", cnn.loss)
acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

# Train Summaries
train_summary_op = tf.merge_summary([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

# Dev summaries
dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
```

`merge_summary`可以将多个summary操作合并为一个。

### 12. 训练模型 - 检查点（Checkpointing）

保存模型的参数以免之后重载使用。可以使用`Early stopping`技术选择最好的参数。

```python
# Checkpointing
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# Tensorflow assumes this directory already exists so we need to create it
if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())
```

### 13. 实例化变量

```python
sess.run(tf.initialize_all_variables())
```

也可以分别调用变量的初始化函数，当想使用已经训练好的词向量时可以使用这种方式，否则所有的都将被初始化为默认值。

### 14.定义单次训练、评估、参数更新

```python
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
    cnn.input_x: x_batch,
    cnn.input_y: y_batch,
    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
    feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, 			accuracy))
    train_summary_writer.add_summary(summaries, step)
```

### 15.定义单词在测试集上的测试

```python
def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
    cnn.input_x: x_batch,
    cnn.input_y: y_batch,
    cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
    feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, 		accuracy))
    if writer:
    writer.add_summary(summaries, step)
```

### 16.定义训练循环

在数据的不同组合（batches）上训练模型，调用  `train_step`  函数，随时评估模型，并应用检查点保存参数

```python
# Generate batches
batches = data_helpers.batch_iter(
zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch…
for batch in batches:
x_batch, y_batch = zip(*batch)
train_step(x_batch, y_batch)
current_step = tf.train.global_step(sess, global_step)
if current_step % FLAGS.evaluate_every == 0:
    print("\nEvaluation:")
    dev_step(x_dev, y_dev, writer=dev_summary_writer)
    print("")
if current_step % FLAGS.checkpoint_every == 0:
    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
    print("Saved model checkpoint to {}\n".format(path))
```

### 17.使用TensorBoard进行数据可视化

```bash
tensorboard –logdir /PATH_TO_CODE/runs/1449760558/summaries/
```

Tips:

- 使用的batch_size太小，会导致训练结果曲线不光滑
- 测试集上的正确率显著低于训练集，是因为数据集太小，更强的正则化，更少的模型参数将可能解决这个问题。
- 刚开始训练集结果度量明显不如测试集，是因为施加了dropout,而开始时模型训练的并不好，还被砍去了一些功能orz/
-  Constrain the L2 norm of the weight vectors in the last layer, just like the [original paper](http://arxiv.org/abs/1408.5882). You can do this by defining a new operation that updates the weight values after each training step. 
-  Add L2 regularization to the network to combat overfitting, also experiment with increasing the dropout rate. (The code on Github already includes L2 regularization, but it is disabled by default) 
- 







