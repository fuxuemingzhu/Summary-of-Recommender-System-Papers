# 推荐系统论文归类总结

标签（空格分隔）： 推荐系统 论文

---

本文主要记录较新的推荐系统论文，并对类似的论文进行总结和整合。
1.综述
1.《Deep Learning based Recommender System: A Survey and New Perspectives》
2.《Use of Deep Learning in Modern Recommendation System: A Summary of Recent Works》
这几个综述比较全面，介绍了目前的深度学习在推荐系统中的应用。
2.基于自编码器
1.《AutoRec: Autoencoders Meet Collaborative Filtering》
2.《Training Deep AutoEncoders for Collaborative Filtering》NVIDIA的文章，偏向于工程实现
3.《Deep Collaborative Autoencoder for Recommender Systems:A Unified Framework for Explicit and Implicit Feedback》
4.《Collaborative Denoising Auto-Encoders for Top-N Recommender Systems》对推荐系统的归纳很好，公式很详细。
文章的思想基本一样，本质都是协同过滤。优化的目标在自编码器的基础上稍作修改，优化目标里只去优化有观测值的数据。



3. 基于矩阵分解
1.《Leveraging Long and Short-term Information in Content-aware Movie Recommendation》几个模型的融合
这个文章简直好大全。MF，LSTM，CNN，GAN全都用上了。
本质是学习得到用户和电影的隐层向量表示。学习的方式是最小化能观测到的电影评分预测值和真实评分值的方根误差。即MF的公式是：

另外，矩阵分解不能学到关于时间变化的用户口味的变化，所以本文用到了LSTM。文章整体的架构如下。

4.Item2Vec
1.《Item2Vec: Neural Item Embedding for Collaborative Filtering》微软的开创性的论文，提出了Item2Vec，使用的是负采样的skip-gram
2.《Item2Vec-based Approach to a Recommender System》给出了开源实现，使用的是负采样的skip-gram
3.《From Word Embeddings to Item Recommendation》使用的社交网站历史check-in地点数据预测下次check-in的地点，分别用了skip-gram和CBOW
固定窗口的skip-gram的目标是最大化每个词预测上下文的总概率：

使用shuffle操作来让context包含每个句子中所有其他元素，这样就可以使用定长的窗口了。

5.上下文感知模型
1.《A Context-Aware User-Item Representation Learning for Item Recommendation》
这个文章提出，以前的模型学到的用户和物品的隐层向量都是一个静态的，没有考虑到用户对物品的偏好。本文提出了上下文感知模型，使用用户的评论和物品总评论，通过用户-物品对进行CNN训练，加入了注意力层，摘要层，学习到的是用户和物品的联合表达。更倾向于自然语言处理的论文，和传统的推荐模型差距比较大。

6.基于视觉的推荐
1.《Telepath: Understanding Users from a Human Vision Perspective in Large-Scale Recommender System》京东最近公开的推荐系统，通过研究商品的封面对人的影响进行推荐
这个文章参考大脑结构，我们把这个排序引擎分为三个组件：一个是视觉感知模块（Vision Extraction），它模拟人脑的视神经系统，提取商品的关键视觉信号并产生激活；另一个是兴趣理解模块（Interest Understanding），它模拟大脑皮层，根据视觉感知模块的激活神经元来理解用户的潜意识（决定用户的潜在兴趣）和表意识（决定用户的当前兴趣）；此外，排序引擎还需要一个打分模块（Scoring），它模拟决策系统，计算商品和用户兴趣（包括潜在兴趣和当前兴趣）的匹配程度。
兴趣理解模块收集到用户浏览序列的激活信号后，分别通过DNN和RNN，生成两路向量。RNN常用于序列分析，我们用来模拟用户的直接兴趣，DNN一般用以计算更广泛的关系，用来模拟用户的间接兴趣。最终，直接兴趣向量和间接兴趣向量和候选商品激活拼接在一起，送往打分模块。打分模块是个普通的DNN网络，我们用打分模块来拟合用户的点击/购买等行为。最终这些行为的影响通过loss回馈到整个Telepath模型中。在图右侧，还引入了类似Wide & Deep网络的结构，以增强整个模型的表达能力。

7.基于RNN的推荐
1.《Session-based Recommendations with Recurrent Neural Networks》 2016年的文章，GRU4Rec，使用每个会话中用户的行为记录进行训练。
2.《Recurrent Neural Networks with Top-k Gains for Session-based Recommendations》2018年的新文章，对上文进行了优化；原理相同的
基于RNN的推荐也是源于一个朴素的假设：对于用户的行为序列，相邻的元素有着相近的含义。这种假设适合基于会话的推荐系统，如一次电子商务的会话，视频的浏览记录等。相对于电影推荐，基于会话的推荐系统跟看中短期内用户的行为。
论文想法在于把一个 session 点击一系列 item 的行为看做一个序列，用来训练一个 RNN 模型。在预测阶段，把 session 已知的点击序列作为输入，用 softmax 预测该session下一个最有可能点击的item。
这个文章里用的是GRU，目标是优化pair-wise rank loss。
有一个不错的论文解读文章：http://www.cnblogs.com/daniel-D/p/5602254.html

8.基于图的推荐
1.《Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time》社交网站的图推荐，2017年
本文介绍了 Pinterest 的 Pixie 系统，主要针对他们开发的随机游走和剪枝算法，此外系统本身基于 Stanford Network Analysis Platform 实现。
9.别的公司的推荐系统的发展历程
1.《Related Pins at Pinterest: The Evolution of a Real-World Recommender System》Pinterest的推荐系统发展历程
这个推荐系统主要用到的是随机游走的图算法，Pin2Vec，Learning to Rank等方法。只介绍了思想，没有公司和算法。可以直接看解读：http://blog.csdn.net/smartcat2010/article/details/75194918
1.2013年的时候，推荐系统主要基于Pin-Board的关联图，两个Pin的相关性与他们在同一个Board中出现的概率成正比。
2. 在有了最基本的推荐系统后，对Related Pin的排序进行了初步的手调，手调信号包括但不局限于相同Board中出现的概率，两个Pin之间的主题相似度，描述相似度，以及click over expected clicks得分。
3. 渐渐地，发现单一的推荐算法很难满足产品想要优化的不同目标，所以引入了针对不同产品需求生成的候选集(Local Cands)，将排序分为两部分，机器粗排，和手调。
4. 最后，引入了更多的候选集，并且提高了排序部分的性能，用机器学习实现了实时的个性化推荐排序。

10.数据集
1.《Indian Regional Movie Dataset for Recommender Systems》提供了印度本土的电影观看数据集




