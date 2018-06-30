# sparkml-doc
Documentation for Spark machine learning library
====

## Spark机器学习库(MLlib)指南

MLlib是Spark里的机器学习库。它的目标是使实用的机器学习算法可扩展并容易使用。它提供如下工具：

* 机器学习算法：常规机器学习算法包括分类、回归、聚类和协同过滤。

* 特征工程：特征提取、特征转换、特征选择以及降维。

* 管道：构造、评估和调整的管道的工具。

* 存储：保存和加载算法、模型及管道

* 实用工具：线性代数，统计，数据处理等。

＊注意：主要接口为基于数据框的接口，基于RDD的接口目前处于维护模式。

在Spark2.0中，spark.mllib包中的RDD接口已进入维护模式。现在主要的机器学习接口为spark.ml包中的基于数据框接口。

## 目录

### 1. [管道](ml-pipeline.md)

#### 1.1 [DataFrame](ml-pipeline.md#dataframe)

#### 1.2 [变换器](ml-pipeline.md#transformers)

#### 1.3 [估计器](ml-pipeline.md#estimators)

#### 1.4 [管道原理](ml-pipeline.md#pipeline)

#### 1.5 [参数](ml-pipeline.md#parameters)

### 2. 特征提取、特征变换、特征选择

#### 2.1 特征提取

    2.1.1 词频－逆向文件频率（TF-IDF）

    2.1.2 Word2Vec

    2.1.3 计数向量器

#### 2.2 特征变换

    2.2.1 分词器

    2.2.2 停用词移除

    2.2.3 n-gram

    2.2.4 二值化

    2.2.5 主成分分析（PCA）

    2.2.6 多项式展开

    2.2.7 离散余弦变换（DCT）

    2.2.8 字符串－索引变换

    2.2.9 索引－字符串变换

    2.2.10 独热编码

    2.2.11 向量－索引变换

    2.2.12 正则化

    2.2.13 标准缩放

    2.2.14 最大值－最小值缩放

    2.2.15 最大值－平均值缩放

    2.2.16 离散化重组

    2.2.17 元素乘积

    2.2.18 SQL转换器

    2.2.19 向量汇编

    2.2.20 分位数求解器

#### 2.3 特征选择

    2.3.1 向量机

    2.3.2 R公式

    2.3.3 选择

### 3 [分类和回归](ml-classification-regression.ipynb)

#### 3.1 [分类](ml-classification-regression.ipynb#分类)

    3.1.1 [logistic回归](ml-classification-regression.ipynb#logistic回归)

    3.1.2 [决策树分类器](ml-classification-regression.ipynb#决策树分类器)

    3.1.3 [随机森林分类器](ml-classification-regression.ipynb#随机森林分类器)

    3.1.4 [梯度提升树分类器](ml-classification-regression.ipynb#梯度提升树分类器)

    3.1.5 [多层感知机](ml-classification-regression.ipynb#多层感知机)

    3.1.6 [线性支持向量机](ml-classification-regression.ipynb#线性支持向量机)

    3.1.7 [一对多分类器](ml-classification-regression.ipynb#一对多分类器)

    3.1.8 [朴素贝叶斯](ml-classification-regression.ipynb#naive-bayes)

#### 3.2 [回归](ml-classification-regression.ipynb#回归)

    3.2.1 [线性回归](ml-classification-regression.ipynb#线性回归)

    3.2.2 [广义线性模型](ml-classification-regression.ipynb#广义线性模型)

    3.2.3 [回归树](ml-classification-regression.ipynb#回归树)

    3.2.4 [随机森林回归](ml-classification-regression.ipynb#随机森林回归)

    3.2.4 [梯度提升树回归](ml-classification-regression.ipynb#梯度提升树回归)

    3.2.5 [生存回归](ml-classification-regression.ipynb#生存回归)

    3.2.6 [保序回归](ml-classification-regression.ipynb#保序回归)

#### 3.3 实现细节

    3.3.1 [线性方法](ml-classification-regression.ipynb#线性方法)

    3.3.2 [决策树](ml-classification-regression.ipynb#决策树)

    3.3.3 [树集成](ml-classification-regression.ipynb#树集成)

    3.3.4 [随机森林](ml-classification-regression.ipynb#随机森林)

    3.3.5 [梯度提升树](ml-classification-regression.ipynb#梯度提升树)

### 4 [聚类](ml-clustering.ipynb)

#### 4.1 [K均值聚类](ml-clustering.ipynb#K均值)

#### 4.2 [隐Dirichlet分配（LDA）](ml-clustering.ipynb#隐Dirichlet分配（LDA）)

#### 4.3 [二分K均值](ml-clustering.ipynb#二分K均值)

#### 4.4 [高斯混合模型（GMM）](ml-clustering.ipynb#高斯混合模型（GMM）)

### 5 协同过滤

### 6 模型选择和调试

#### 6.1 交叉检验

#### 6.2 训练检验分裂

### 7 高级主题
