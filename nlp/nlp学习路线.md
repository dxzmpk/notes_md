# nlp学习路线

## 文本分类

1. 文本分类算法总结 [链接](https://monkeylearn.com/text-classification/#machine-learning-based-systems)

    ![Deep Learning vs Traditional Machine Learning algorithms](https://monkeylearn.com/static/img/text-classification-deep-learning-data.png) 

   这张图显示的是，随着训练数据的增加，传统的机器学习方法会遇到瓶颈，而深度学习分类器则不然。

   [GloVe网址](https://nlp.stanford.edu/projects/glove/)

   

2. 文本分类数据集

   主题分类：

   - [路透社新闻数据集](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html)：可能是文本分类中使用最广泛的数据集，其中包含来自路透社的21,578条新闻文章，按其主题分别标记了135个类别，例如政治，经济，体育和商业。
   - [20个新闻组](http://qwone.com/~jason/20Newsgroups/)：另一个受欢迎的数据集，包含20个不同主题的约20,000个文档。

   情绪分析：

   - [亚马逊商品评论](http://jmcauley.ucsd.edu/data/amazon/)：一个知名的数据集，包含约1.43亿条评论和1996年5月至2014年7月的星级（1-5星）。您可以在此处获取亚马逊商品评论的替代数据集。
   - [IMDB评论](http://ai.stanford.edu/~amaas/data/sentiment/)：要小得多的数据集，其中包含来自互联网电影数据库（IMDB）的25,000条电影评论分别为正面和负面的评论。
   - [Twitter航空公司情绪](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)：此数据集包含约15,000条关于被标记为正面，中性和负面的航空公司的推文。

   其他流行的数据集：

   - [Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase)：包含4,601封电子邮件的标签为垃圾邮件和非垃圾邮件的数据集。
   - [SMS垃圾邮件收集](https://www.kaggle.com/uciml/sms-spam-collection-dataset)：另一个用于检测垃圾邮件的数据集，包含5574条标记为垃圾邮件或合法的SMS消息。
   - [仇恨言论和攻击性语言](https://github.com/t-davidson/hate-speech-and-offensive-language)：此数据集包含24,802个带标签的推文，分为三类：干净，仇恨言论和攻击性语言。

3. 文本分类库

   ##### 使用Python进行文本分类

   对于需要在机器学习模型中工作的开发人员和数据科学家而言，Python通常是首选的编程语言。简单的语法，庞大的社区以及其数学库的科学计算友好性是Python在该领域如此盛行的一些原因。

   [Scikit-learn](http://scikit-learn.org/)是通用机器学习的入门库之一。它支持多种算法，并提供简单有效的功能来处理文本分类，回归和聚类模型。如果您是机器学习的初学者，则scikit-learn是最友好的文本分类入门库之一，它在网络上提供了许多教程和分步指南。

   [NLTK](https://www.nltk.org/)是一个流行的图书馆，专注于[自然语言处理](https://monkeylearn.com/blog/definitive-guide-natural-language-processing/)（NLP），背后有很大的社区。它为文本分类提供了极大的便利，因为它提供了各种有用的工具来使机器理解文本，例如将段落拆分为句子，拆分单词以及识别这些单词的语音部分。

   [SpaCy](https://spacy.io/)是一种现代而更新的NLP库，它是一种工具包，其方法比NLTK更为简洁明了。例如，spaCy仅实现单个词干分析器（NLTK具有9个不同的选项）。SpaCy还集成了[单词嵌入功能](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/)，可以帮助提高文本分类的准确性。

   一旦准备好尝试更复杂的算法，就应该检查一下Keras，TensorFlow和PyTorch等深度学习库。[Keras](https://keras.io/)可能是最好的起点，因为它旨在简化循环神经网络（RNN）和卷积神经网络（CNN）的创建。[TensorFlow](https://www.tensorflow.org/)是用于实施深度学习算法的最受欢迎的开源库。该库由Google开发，并由Dropbox，eBay和Intel等公司使用，该库经过优化，可用于设置，训练和部署具有大量数据集的人工神经网络。尽管比Keras难掌握，但它是深度学习领域无可争议的领导者。TensorFlow的可靠替代品是[PyTorch](https://pytorch.org/)，这是一个广泛的深度学习库，主要由Facebook开发，并得到Twitter，Nvidia，Salesforce，斯坦福大学，牛津大学和Uber的支持。

   ##### Java文本分类

   广泛用于实现机器学习模型的另一种编程语言是Java。像Python一样，它拥有一个庞大的社区，一个广泛的生态系统以及大量用于机器学习和NLP的开源库。

   [CoreNLP](https://stanfordnlp.github.io/CoreNLP/)是Java中最流行的NLP框架。它由斯坦福大学创建，提供了多种用于理解人类语言的工具，例如文本解析器，词性（POS）标记器，命名实体识别器（NER），共指解析系统和信息提取工具。 。

   另一种流行的用于自然语言任务的工具包是[OpenNLP](https://opennlp.apache.org/)。它是由Apache Software Foundation创建的，提供了许多语言分析工具，可用于文本分类，例如标记化，句子分段，词性标记，分块和解析。

   [Weka](https://www.cs.waikato.ac.nz/ml/weka/)是由怀卡托大学开发的机器学习库，其中包含许多工具，例如分类，回归，聚类和数据可视化。它提供了一个图形用户界面，用于将Weka的算法集合直接应用于数据集，以及一个API，可以从您自己的Java代码中调用这些算法。

   
   
   

## 文本分类方法概述 [链接](https://pdf.sciencedirectassets.com/271506/1-s2.0-S0957417418X00104/1-s2.0-S095741741830215X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGcaCXVzLWVhc3QtMSJHMEUCIG8IUbUw0BfTL8vyto16k5bPtIXQxZH%2BC32gTOPnjLsvAiEAmbAtDuDhfAuIMrPwKkZG5RvAN2vP2lrtG%2Bt%2FlLMfimEq2QIIgP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgwwNTkwMDM1NDY4NjUiDNIMejW0JU%2F%2F8OeW1yqtAuOSzcH2NYkqIkWSXCE%2FK1v79OUGWi9Wme1DaRkLB0xBiYtOb23r19%2FvboIIAYbX91Af%2FGKffEmBnj55%2B35419sqCD2IEfn55nYi3lr%2FlOvk5PQ4ks6F6xUMBBq%2BMgNh2P9O2nYn2zgWGtTH4zbfx3l8do%2FEdJKWN1VzoUe86MLElUjY9rmXd5SABc8gtmyBnrTMUaAVdFfrJyiGuLOdndJ1iTxZkOm1d6IXokGjWUSBzSfZjDX%2Fz16t9vcc67eDJDp9P2Aa1u4%2FHYx%2F49dTsA0dRwsPqnr4xeHoaVFtu0ERhkU6Ly%2BNkbjrLkm7y2b%2FPZqUHSX2YqG4k44MpBViLcmog3O2%2FYM2XaBAmdobQiSc7Av6M6gDbVlgTgse3r4vJCpqcba%2FO0aklR0zgLEw85eN7gU6zwL%2BLPJRp9npOdxJc%2FgBAkpuxAoy4XPfG3%2F4teFxWe6wF1z5jZ0uRAmC8ikZlZnPxbM1h023ONQPalmJk%2BqRCFkYn3O%2BnPZQDjLESVzFaLXC12tSv4WLy8YU%2FqycBmWGVQRlqnPmBfQz6CQOZtEOx1362WG0abhIsmlXwmvtg1ihsDzt4lgydJGB8l0dlQUFkM%2BI2W3RaFJx7nHWxQywg0Lfs5XfUE4p%2BDEjfnJTd64Ey8qMLU%2BXxnVvR%2Fs%2FiSnA4r6DViNSjIj9%2BRINyBVtu30ZDJGIUvnmKPpzfBtJ4yyIu60nQjAHfP1zAACHslRiqqIwEpjvY1Lweus58cwE1D43Kj1CDsjo13ecR%2B5ZQsIkSyUREMepWWWdskKyPws%2Bm8kifgqknIZlmx7U6nPXuwKdHt7PRI6MrmeO5us6eqfgimby8mHNPwnyzf4rb9Cz9Q%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191107T000619Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY52B2GHTO%2F20191107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=11fa009789d5d3d1ab1963854bdd7872c697bb9670da042816e1e381d5141bc8&hash=4f4e9a67ef6f0170acfe19206f281e674142d6c402bffac6b636122d39152d3c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S095741741830215X&tid=spdf-b615ddc1-6080-46ca-a4ff-49d6812d911f&sid=0a81af724adee644f49a341865e65b0b1ee1gxrqa&type=client)

### 文本分类的过程概述

**![1573091828712](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1573091828712.png)**

The investigated baseline process for text classification includes the six elements mentioned in Section 2. Fig. 1 presents a flowchart of this process, which we discuss briefly below. As we can see in Fig. 1, the classification process starts with data acquisition from various text sources, including internal datasets, the Internet, and open databases. From the data acquisition, we obtain a dataset representing a physical or business process. Next, the dataset is pre-processed to generate a representation required by the selected learning method.然后，数据集被预处理为学习方法所需的数据表示。 This non-trivial issue consists of two phases. First, the features are constructed from the data. Then, they are accordingly weighted with the selected feature representation algorithm to yield the appropriate data representation.这个不平凡的工作包括两部分，首先从数据中构造特征，然后加权求和。 Then, the number of features are reduced by the feature selection method. Subsequently, the reduced features are projected onto a lower dimensionality to achieve the optimal data representation.其次，选择的属性被投影到更低的维度，来实现更加优化的数据表示。 Following this, different learning approaches are used to train a classification function to recognise a target concept. 紧接着，使用不同的学习算法，训练一个分类函数来识别目标概念。When a classification model is adequately developed, it can classify incoming data that have to be represented in a manner similar to in the training phase. 在分类模型被训练好后，它可以被用来分类输入的数据（数据需以训练阶段的表示相同）Consequently, the classifier produces a decision that defines the class of each input vector.也可以说，分类器产生了一个决策函数，可以把向量定义为相应的类。 Technically, the decisions are probabilities or weights. 事实上，这个决策是概率或者是权重。Finally, the evaluation procedure is utilised to estimate the text classification process operation. All the above phases are the main elements of the framework for text classification. 

### 文本分类的不同研究领域和方法

 Text document classification (text classification) is the problem of assigning predefined classes (labels) to an unlabelled text document. Numerous studies present different approaches and applications of text classification. The various categories of text classification are domain, classification purpose, classification task, general approach, and dedicated approach. Table 3 presents some details about these categories. 

![1573093292201](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1573093292201.png)

# ULMFiT 

For NLP classification the current state of the art approach is *Universal Language Model Fine-tuning* (ULMFiT). ULMFiT is an effective transfer learning method that can be applied to any task in NLP, but at this stage we have only studied its use in classication tasks. The approach is described and analyzed in the [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) paper by fast.ai’s Jeremy Howard and [Sebastian Ruder](http://ruder.io/) from the NUI Galway Insight Centre. 

To learn to use ULMFiT and access the open source code we have provided, see the following resources:

- ULMFiT is discussed in depth in [lesson 10](http://course.fast.ai/lessons/lesson10.html) of fast.ai’s [Cutting Edge Deep Learning for Coders](http://course.fast.ai/part2.html). A gentler introduction is available in [lesson 4](http://course.fast.ai/lessons/lesson4.html) of [Practical Deep Learning for Coders](http://course.fast.ai/)
- The [fastai library](https://github.com/fastai/fastai) provides modules necessary to train and use ULMFiT models. In particular, you will want to use `fastai.text` and `fastai.lm_rnn`
- The scripts used for the ULMFiT paper are available in the [imdb_scripts](https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts) folder in the fastai repository.
- The pre-trained Wikitext 103 model and vocab are [available here](http://files.fast.ai/models/wt103/)
- The paper and code are being discussed in the [fast.ai discussion forums](http://forums.fast.ai/c/part2-v2/). Feel free to join the discussion!

## Transformer

![seq to seq](https://guillaumegenthial.github.io/assets/img2latex/seq2seq_vanilla_encoder.svg)

![seq to seq](https://guillaumegenthial.github.io/assets/img2latex/seq2seq_vanilla_decoder.svg)

1. Seq-seq[模型](https://www.analyticsvidhya.com/blog/2018/03/essentials-of-deep-learning-sequence-to-sequence-modelling-with-attention-part-i/?utm_source=blog&utm_medium=understanding-transformers-nlp-state-of-the-art-models), [图片来自于链接](https://guillaumegenthial.github.io/sequence-to-sequence.html)

    *Intuitively, the hidden vector represents the “amount of meaning” that has not been decoded yet.* 

   ```
   Joe went to the kitchen. Fred went to the kitchen. Joe picked up the milk.
   Joe travelled to the office. Joe left the milk. Joe went to the bathroom.
   ```

   问题是， *Where was Joe before the office?* 

   这里的难点是，系统需要理解两个维度：

   - 英语语言的机理和单词的序列
   - 围绕陈述中人员的事件序列

   这可以看做是一个序列建模问题。这张图是多输入-单输出问题。

   序列建模问题还包括多输入-多输出问题：

   - 机器翻译
   - 人机对话系统
   - 视频信息捕捉：对每一帧视频产生描述

   广义上，编码器网络的任务是了解输入序列，并为其创建较小的尺寸表示。 然后将该表示转发到解码器网络，该解码器网络生成自己的表示输出的序列。

   例：使用RNN序列模型实现机器翻译的细节：

   - Encoder和Decoder都是RNN
   - Encoder中的每一步，都从输入序列中得到一个词向量和之前的一个隐状态
   - 隐藏状态在每个时间点都升级一些
   - 最后一个单元的隐状态就是全文向量，它包含了输入序列的信息
   - 这个全文向量输入到decoder中去构建目标序列
   - 如果使用注意力机制，最终被传入decoder的向量就是**隐向量的加权总和**

2.  Attention机制：

      ![](https://guillaumegenthial.github.io/assets/img2latex/seq2seq_attention_mechanism_new.svg)

      Attention is a mechanism that forces the model to learn to focus (=to attend) on specific parts of the input sequence when **decoding**, instead of relying only on the hidden vector of the decoder’s LSTM.  

3. 集束搜索（beam search）

   集束搜索可以认为是维特比算法的贪心形式，在维特比所有中由于利用动态规划导致当字典较大时效率低，而集束搜索使用beam size参数来限制在每一步保留下来的可能性词的数量。集束搜索是在测试阶段为了获得更好准确性而采取的一种策略，在训练阶段无需使用。 

   ​		集束搜索属于贪心算法，然而考虑的搜索空间更大（考虑接下来的k个单词，然后找出综合概率最大的结果），而采用一个相对的较优解。而维特比算法能在词典大小比较小时快速找到全局最优解。

   

4. seq2seq模型的限制

   - 在处理长文本时依然是个挑战
   - 因为是序列性质，所以模型在计算时无法实现并行处理，这些挑战由Transformer解决。

5. Transformer基本概念

   特征：

   -  eschewing recurrence 
   - relying entirely on an attention mechanism to draw global dependencies between input and output

## 迁移学习 [链接](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

 Transfer learning only works in deep learning if the model features learned from the first task are general. 

### Develop Model Approach

1. **Select Source Task**. You must select a related predictive modeling problem with an abundance of data where there is some relationship in the input data, output data, and/or concepts learned during the mapping from input to output data.
2. **Develop Source Model**. Next, you must develop a skillful model for this first task. The model must be better than a naive model to ensure that some feature learning has been performed.
3. **Reuse Model**. The model fit on the source task can then be used as the starting point for a model on the second task of interest. This may involve using all or parts of the model, depending on the modeling technique used.
4. **Tune Model**. Optionally, the model may need to be adapted or refined on the input-output pair data available for the task of interest.

### Pre-trained Model Approach

1. **Select Source Model**. A pre-trained source model is chosen from available models. Many research institutions release models on large and challenging datasets that may be included in the pool of candidate models from which to choose from.
2. **Reuse Model**. The model pre-trained model can then be used as the starting point for a model on the second task of interest. This may involve using all or parts of the model, depending on the modeling technique used.
3. **Tune Model**. Optionally, the model may need to be adapted or refined on the input-output pair data available for the task of interest.

This second type of transfer learning is common in the field of deep learning.

### Transfer Learning with Language Data

It is common to perform transfer learning with natural language processing problems that use text as input or output.

For these types of problems, a word embedding is used that is a mapping of words to a high-dimensional continuous vector space where different words with a similar meaning have a similar vector representation.

Efficient algorithms exist to learn these distributed word representations and it is common for research organizations to release pre-trained models trained on very large corpa of text documents under a permissive license.

Two examples of models of this type include:

- [Google’s word2vec Model](https://code.google.com/archive/p/word2vec/)
- [Stanford’s GloVe Model](https://nlp.stanford.edu/projects/glove/)

These distributed word representation models can be downloaded and incorporated into deep learning language models in either the interpretation of words as input or the generation of words as output from the model.

In his book on Deep Learning for Natural Language Processing, Yoav Goldberg cautions:

> … one can download pre-trained word vectors that were trained on very large quantities of text […] differences in training regimes and underlying corpora have a strong influence on the resulting representations, and that the available pre-trained representations may not be the best choice for [your] particular use case.

— Page 135, [Neural Network Methods in Natural Language Processing](http://amzn.to/2fwTPCn), 2017.

## cs224n6 Model and Recurrent neural network

### 传统的模型

- 很高的RAM要求

### 语言模型的定义

- 给定几个词，产生下一个词的概率是多少
- 产生一个句子片段的概率

**N-GRAM**

统计n-gram来构建语言模型
$$
P(x^{(t+1)}\mid x^{(t)},...,x^{(1)}) = \\ \\
P(x^{(t+1)}\mid x^{(t)},...,x^{(t-n+2)})=\\ \\
\frac{P(x^{(t+1)},x^{(t)},...,x^{(t-n+2)})}{P(x^{(t)} ,...,x^{(t-n+2)})}
$$
 存在的问题：

- 可能会忽略上下文信息，导致下一个词的预测不准确。
- 稀疏问题，如果分子为0，则整个概率为0。即如果在训练语料中没有出现某个句子，就认为这个句子不可能出现，这是不对的。 通过平滑技术来解决。
- 分母为0，可以通过减小n来解决

**A fixed-window neural Language Model**

![1574061943982](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574061943982.png)

假设窗口尺寸为4

![1574062139921](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574062139921.png)

优点：

- 不存在稀疏问题
- 不需要存储所有的n-gram, 只需要存储word-embedding表

缺点：

- 在每一次学习中，一个位置的单词只会对一部分的权重矩阵产生影响，这会导致最终学习到的 结果存在很多的重复，导致结果的冗余。
- window size不可变，而上一个问题也是由于window_size不可变造成的，基于此提出了基于RNN的语言模型

**RNN language model**

![1574063874614](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574063874614.png)

优点：

- 可以计算任何尺寸的输入
- 第t步的计算可以使用多步之前的信息
- 当输入长度增加时，模型的复杂程度并不增加。

缺点：

- 需要一步一步计算，所以会比较慢
- 难以记住很多步之前的信息--记忆易丢失

训练过程：

在每一步都预测当前的输出，y_hat,和one-hot向量作比较计算cross - entropy
$$
J^{(t)}(\theta) = CE(y^{(t)},\hat{y}^{(t)}) = -\sum_{w\epsilon V}y^{(t)}_w log\;\hat{y}^{(t)}_w = -log\;\hat{y}^{(t)}_{x_{t+1}}
$$
对于整个训练集，计算平均误差得到总误差

![1574066893055](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574066893055.png)

结果度量：

使用perplexity(困惑度)，公式如下：
$$
perplexity = \prod_{t=1}^T(\frac{1}{P_{LM}(x^{(t+1)}\mid x^{(t)},...,x^{(1)})})^{\frac{1}{T}}
$$
困惑度越低越好

## cs224n 7 more neural network

为何梯度消失是一个问题：

![1574302023279](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574302023279.png)

LSTM的结构

![1574305388242](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574305388242.png)

![1574305406710](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574305406710.png)

WMT LSTM

GRU -- simpler, less parameters, faster training

![1574321408090](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574321408090.png)

Bi-directional RNN

双向RNN，只有当获取了全部的文本之后才可以进行。

## cs224n 8 Translation, Seq2Seq, Attention

$$
argmax_yP(x\mid y)P(y)
$$

左侧为翻译模型，主要模拟单词和句子应如何被翻译。(如何确定其对应关系，训练过程需要大量的包含句子之间对应关系的语料)

右侧为语言模型，主要告诉我们什么样的句子是流利的语言。(可以通过语言模型来衡量（这里只需要大量的目标语言即可)）

这里的操作，实际上是把原来需要一步完成的工作拆成两部分来完成，一次关注翻译的对应关系，另一次关注流利程度。

问题：如何学习翻译模型？

采用同样的方法，将任务分成两部分。第一部分学习词和词之间的对应关系，第二部分学习由给定的词如何学习得到目标词汇。 

NMT 使用一个神经网络进行机器翻译

Seq-Seq

![1574409489011](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574409489011.png)

Seq2Seq是Conditional Language Model 模型的一个例子

Language Model: 模型预测目标句子的下一个输出

Conditional: 因为其预测是基于源句子的条件概率

![1574409993096](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574409993096.png)

**Beam Search Decoding**

Search Spcae: k

跟踪长度为k的搜索空间

比exhaustive search更有效率

![1574413077406](C:\Users\YOGA710\AppData\Roaming\Typora\typora-user-images\1574413077406.png)

































































## LSA [wiki](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

构建此项、文档矩阵之后，需要对矩阵进行降维，降维的原因是：

- 词项、文档矩阵太稀疏，需要计算资源过多，所以通过近似只保留最关键的信息
- 初始的矩阵噪声太多，例如，存在一些实例应该被消除，从这个方面来看，近似矩阵可以被成为去噪矩阵。

 LSA是一个完全自动化的过程，用来抽取和推测单词的上下文含义，它不需要任何人们已经建立的知识，语法等，而只需要输入纯文本。‘

### 第一步，词-矩阵表示

其中的每一个值代表word’s importance in the particular passage and the degree to which the word type
carries information in the domain of discourse in general

### 第二步，使用SVD

将原来的矩阵解构为三个矩阵的乘积，分别代表的含义如下：

```
One component matrix describes the original row entities as vectors ofderived orthogonal factor values
another describes the original column entities in the same way
the third is a diagonal matrix containing scaling values such that when the three components are matrix-multiplied, the original matrix is reconstructed. 
```

我们可以通过减少对角矩阵中的特征值来减少结果的维度。

处于计算原因，对于很大的语料库，我们构建的矩阵数目会被限制在几千左右

### 第三步，选择几个维度重构矩阵

LSA通过增大或者减小键值来满足数据之间的相互限制关系。





