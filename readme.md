# 推荐系统论文归类总结

本文主要记录较新的推荐系统论文，并对类似的论文进行总结和整合。

## 综述

1. 《Deep Learning based Recommender System: A Survey and New Perspectives》
2. 《Use of Deep Learning in Modern Recommendation System: A Summary of Recent Works》

这几个综述比较全面，介绍了目前的深度学习在推荐系统中的应用。

## 基于协同过滤

1.《Amazon.com Recommendations Item-to-Item Collaborative Filtering》亚马逊发表的基于物品的协同过滤

在 User-based 方法中，随着用户数量的不断增多，在大数量级的用户范围内进行“最近邻搜索”会成为整个算法的瓶颈。Item-based 方法通过计算项之间的相似性来代替用户之间的相似性。对于项来讲，它们之间的相似性要稳定很多，因此可以离线完成工作量最大的相似性计算步骤，从而大大降低了在线计算量，提高推荐效率。

在 Item-based 方法中，要对 A 和 B 进行项相似性计算，通常分为两步：1）找出同时对 A 和 B 打过分的组合；2）对这些组合进行相似度计算，常用的算法包括：皮尔森相关系数、余弦相似性、调整余弦相似性和条件概率等。伪代码如下：

![此处输入图片的描述][1]

2.《Item-Based Collaborative Filtering Recommendation Algorithms》影响最广的，被引用的次数也最多的一篇推荐系统论文。

文章很长，非常详细地探讨了基于Item-based 方法的协同过滤，作为开山之作，大体内容都是很基础的知识。文章把Item-based算法分为两步：

（1）相似度计算，得到各item之间的相似度

- 基于余弦（Cosine-based）的相似度计算
- 基于关联（Correlation-based）的相似度计算
- 调整的余弦（Adjusted Cosine）相似度计算

（2）预测值计算，对用户未打分的物品进行预测

- 加权求和。用户u已打分的物品的分数进行加权求和，权值为各个物品与物品i的相似度，然后对所有物品相似度的和求平均，计算得到用户u对物品i打分。
- 回归。如果两个用户都喜欢一样的物品，因为打分习惯不同，他们的欧式距离可能比较远，但他们应该有较高的相似度 。在通过用线性回归的方式重新估算一个新的R(u,N).

文章很经典，没有太难理解的部分，可以看别人的笔记：[https://blog.csdn.net/huagong_adu/article/details/7362908][2]

## 基于矩阵分解

1.《Matrix Factorization Techniques for Recommender Systems》矩阵分解，推荐系统领域里非常经典、频繁被引用的论文。

这个论文是推荐系统领域第一篇比较正式、全面介绍融合了机器学习技术的矩阵分解算法。矩阵分解是构建隐语义模型的主要方法，即通过把整理、提取好的“用户—物品”评分矩阵进行分解，来得到一个用户隐向量矩阵和一个物品隐向量矩阵。

![此处输入图片的描述][3]

在得到用户隐向量和物品隐向量（如都是2维向量）之后，我们可以将每个用户、物品对应的二维隐向量看作是一个坐标，将其画在坐标轴上。虽然我们得到的是不可解释的隐向量，但是可以为其赋予一定的意义来帮助我们理解这个分解结果。比如我们把用户、物品的2维的隐向量赋予严肃文学（Serious）vs.消遣文学（Escapist）、针对男性（Geared towards males）vs.针对女性（Geared towards females），那么可以形成论文中那样的可视化图片：

![此处输入图片的描述][4]

矩阵分解算法具有的融合多种信息的特点也让算法设计者可以从隐式反馈、社交网络、评论文本、时间因素等多方面来弥补显示反馈信息不足造成的缺陷，可以根据需要很容易的把公式进行改变。比如考虑到时间变化的用户、项目的偏差，可以对预测评分函数改写成：

![此处输入图片的描述][5]

关于这个文章比较详细的解读：[论文篇：Matrix Factorization Techniques for RS][6]，[矩阵分解（MATRIX FACTORIZATION）在推荐系统中的应用][7]。

2.《Leveraging Long and Short-term Information in Content-aware Movie Recommendation》几个模型的融合

这个文章简直好大全。MF，LSTM，CNN，GAN全都用上了。

本质是学习得到用户和电影的隐层向量表示。学习的方式是最小化能观测到的电影评分预测值和真实评分值的方根误差。即MF的公式是：

![此处输入图片的描述][8]

另外，矩阵分解不能学到关于时间变化的用户口味的变化，所以本文用到了LSTM。文章整体的架构如下。

![此处输入图片的描述][9]

## 基于自编码器

1. 《AutoRec: Autoencoders Meet Collaborative Filtering》
2. 《Training Deep AutoEncoders for Collaborative Filtering》NVIDIA的文章，偏向于工程实现
3. 《Deep Collaborative Autoencoder for Recommender Systems:A Unified Framework for Explicit and Implicit Feedback》
4. 《Collaborative Denoising Auto-Encoders for Top-N Recommender Systems》对推荐系统的归纳很好，公式很详细。

这几篇文章的思想基本一样，本质都是协同过滤。优化的目标在自编码器的基础上稍作修改，优化目标里只去优化有观测值的数据。

![此处输入图片的描述][10]

![此处输入图片的描述][11]

![此处输入图片的描述][12]

## Item2Vec

1. 《Item2Vec: Neural Item Embedding for Collaborative Filtering》微软的开创性的论文，提出了Item2Vec，使用的是负采样的skip-gram
2. 《Item2Vec-based Approach to a Recommender System》给出了开源实现，使用的是负采样的skip-gram
3. 《From Word Embeddings to Item Recommendation》使用的社交网站历史check-in地点数据预测下次check-in的地点，分别用了skip-gram和CBOW

固定窗口的skip-gram的目标是最大化每个词预测上下文的总概率：

![此处输入图片的描述][13]

使用shuffle操作来让context包含每个句子中所有其他元素，这样就可以使用定长的窗口了。

![此处输入图片的描述][14]

## 上下文感知模型

1. 《A Context-Aware User-Item Representation Learning for Item Recommendation》

这个文章提出，以前的模型学到的用户和物品的隐层向量都是一个静态的，没有考虑到用户对物品的偏好。本文提出了上下文感知模型，使用用户的评论和物品总评论，通过用户-物品对进行CNN训练，加入了注意力层，摘要层，学习到的是用户和物品的联合表达。更倾向于自然语言处理的论文，和传统的推荐模型差距比较大。

![此处输入图片的描述][15]

## 基于视觉的推荐

1.《Telepath: Understanding Users from a Human Vision Perspective in Large-Scale Recommender System》京东最近公开的推荐系统，通过研究商品的封面对人的影响进行推荐

这个文章参考大脑结构，我们把这个排序引擎分为三个组件：一个是视觉感知模块（Vision Extraction），它模拟人脑的视神经系统，提取商品的关键视觉信号并产生激活；另一个是兴趣理解模块（Interest Understanding），它模拟大脑皮层，根据视觉感知模块的激活神经元来理解用户的潜意识（决定用户的潜在兴趣）和表意识（决定用户的当前兴趣）；此外，排序引擎还需要一个打分模块（Scoring），它模拟决策系统，计算商品和用户兴趣（包括潜在兴趣和当前兴趣）的匹配程度。
兴趣理解模块收集到用户浏览序列的激活信号后，分别通过DNN和RNN，生成两路向量。RNN常用于序列分析，我们用来模拟用户的直接兴趣，DNN一般用以计算更广泛的关系，用来模拟用户的间接兴趣。最终，直接兴趣向量和间接兴趣向量和候选商品激活拼接在一起，送往打分模块。打分模块是个普通的DNN网络，我们用打分模块来拟合用户的点击/购买等行为。最终这些行为的影响通过loss回馈到整个Telepath模型中。在图右侧，还引入了类似Wide & Deep网络的结构，以增强整个模型的表达能力。

![此处输入图片的描述][16]

2.《Visually Explainable Recommendation》可视化地可解释推荐模型

这个文章放在基于视觉的推荐的原因是，比较新奇的地方在于提取了商品封面的特征，并融合到了推荐和推荐解释之中。本文的基础模型使用商品的封面通过预训练好的VGG网络转化为图像向量。对特征进行加权求和之后的结果与商品的向量merge，再与用户的向量内积求总的向量结果，把该结果进行和用户是否购买的真实数据求交叉熵，优化该Loss.文章指出该模型最后训练的结果可以用推荐，也可以用注意力权重来做推荐解释。

本文还提出了进一步的模型Re-VECF。该模型使用商品的用户评论结合图像、用户和商品作单词预测训练GRU。加入用户评论的好处是可以提高推荐的表现、文本评论可能隐含着用户对商品封面重要的偏好。该模型能更好的做出推荐结果和推荐解释。

![此处输入图片的描述][17]

## 基于RNN的推荐

1. 《Session-based Recommendations with Recurrent Neural Networks》 2016年的文章，GRU4Rec，使用每个会话中用户的行为记录进行训练。
2. 《Recurrent Neural Networks with Top-k Gains for Session-based Recommendations》2018年的新文章，对上文进行了优化；原理相同的

基于RNN的推荐也是源于一个朴素的假设：对于用户的行为序列，相邻的元素有着相近的含义。这种假设适合基于会话的推荐系统，如一次电子商务的会话，视频的浏览记录等。相对于电影推荐，基于会话的推荐系统跟看中短期内用户的行为。

论文想法在于把一个 session 点击一系列 item 的行为看做一个序列，用来训练一个 RNN 模型。在预测阶段，把 session 已知的点击序列作为输入，用 softmax 预测该session下一个最有可能点击的item。

这个文章里用的是GRU，目标是优化pair-wise rank loss。

有一个不错的论文解读文章：http://www.cnblogs.com/daniel-D/p/5602254.html

![此处输入图片的描述][18]

## 基于图的推荐

1. 《Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time》社交网站的图推荐，2017年

本文介绍了 Pinterest 的 Pixie 系统，主要针对他们开发的随机游走和剪枝算法，此外系统本身基于 Stanford Network Analysis Platform 实现。

## 基于树的推荐

1. 《Learning Tree-based Deep Model for Recommender Systems》淘宝的推荐系统，2018年最新发布

基于树的推荐是一种比较新奇的一种推荐算法，其设计的目的主要是解决淘宝的巨大的数据问题，给出了一种能线上服务的实时推荐系统的模型。此外，本文证明了此模型在MovieLens-20M和淘宝自己的用户数据上的准确、召回、新奇性都比传统方法好。

采用的数据是隐式反馈，本模型提供几百个候选集，然后实时预测系统会进行排序策略。

树的作用不仅仅是作为索引使用，更重要的是把海量的数据进行了层次化组织。训练过程是如果用户对某个物品感兴趣，那么最大化从该物品节点到根节点的每个节点的联合概率。该路径上的每个节点都和用户有相关性，树的结构从底向上表现出了用户物品的相似性和依赖性。

如下图所示，左侧的三层全连接学习到用户的向量表示，右侧的树结构学到了节点的表示，最后通过二分类来训练出用户是否对该节点感兴趣。训练的损失函数是最小化一个用户对每个采样了的节点的交叉熵。（树结构类似于Hierarchical softmax，也同样使用了负采样等。）

![此处输入图片的描述][19]

## 公司的推荐系统的发展历程

1. 《Related Pins at Pinterest: The Evolution of a Real-World Recommender System》Pinterest的推荐系统发展历程

这个推荐系统主要用到的是随机游走的图算法，Pin2Vec，Learning to Rank等方法。只介绍了思想，没有公司和算法。可以直接看解读：http://blog.csdn.net/smartcat2010/article/details/75194918

> 1. 2013年的时候，推荐系统主要基于Pin-Board的关联图，两个Pin的相关性与他们在同一个Board中出现的概率成正比。
> 2. 在有了最基本的推荐系统后，对Related Pin的排序进行了初步的手调，手调信号包括但不局限于相同Board中出现的概率，两个Pin之间的主题相似度，描述相似度，以及click
> over expected clicks得分。
> 3. 渐渐地，发现单一的推荐算法很难满足产品想要优化的不同目标，所以引入了针对不同产品需求生成的候选集(Local Cands)，将排序分为两部分，机器粗排，和手调。
> 4. 最后，引入了更多的候选集，并且提高了排序部分的性能，用机器学习实现了实时的个性化推荐排序。

![此处输入图片的描述][20]

## 数据集

1. 《Indian Regional Movie Dataset for Recommender Systems》提供了印度本土的电影观看数据集


## 版权声明

本文正在更新中，请谨慎转载。

个人转载请注明作者和仓库地址，商业和自媒体转载前务必联系作者fuxuemingzhu@163.com。


  [1]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p17.png
  [2]: https://blog.csdn.net/huagong_adu/article/details/7362908
  [3]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p14.jpg
  [4]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p15.png
  [5]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p16.png
  [6]: https://zhuanlan.zhihu.com/p/28577447
  [7]: https://blog.csdn.net/houlaizhexq/article/details/39998135
  [8]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p4.png
  [9]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p5.png
  [10]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p1.png
  [11]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p2.png
  [12]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p3.png
  [13]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p6.png
  [14]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p7.png
  [15]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p8.png
  [16]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p9.png
  [17]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p13.png
  [18]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p10.png
  [19]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p12.png
  [20]: https://raw.githubusercontent.com/fuxuemingzhu/Summary-of-Recommender-System-Papers/master/pics/p11.png
