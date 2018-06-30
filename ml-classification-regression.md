---
layout: global
title: 分类和回归
displayTitle: 分类和回归
---

本章节涵盖分类和回归算法，讨论了线性方法、树和集成方法等特定类型的算法。

# 分类

## Logistic回归

Logistic回归是一个流行的分类问题预测方法。它是[广义线性模型](https://en.wikipedia.org/wiki/Generalized_linear_model)的一个特例，预测类别概率。
在`spark.ml`的Logistic回归中二项Logistic回归可用于预测二值输出，而多项Logistic回归可用于预测多值输出。 用`family`参数来在这两算法中选择，也可以不设置它，Spark会推断正确的选择。

  > 把`family`参数设置成`"multinomial"`，多项Logistic回归可用于二分类。它将产生两组系数和截距。

  > 当在有非零常数列的数据集上不带截距拟合`LogisticRegressionModel`时，Spark MLlib对非零常数列输出零系数。这种行为与R glmnet相同但是不同于LIBSVM。

### 二项Logistic回归

关于二项Logistic回归实现的更多背景和细节，请参考[`spark.mllib`中的Logistic回归](mllib-linear-methods.md#logistic-regression)文档。

**样例**

下面的例子展示了如何在二分类问题上训练带elastic net正则化的二项和多项Logistic回归模型。`elasticNetParam`对应于$\alpha$，`regParam`对应于$\lambda$。

关于参数的更多细节可以在[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.LogisticRegression)中找到。

```python
from pyspark.ml.classification import LogisticRegression

# 载入训练数据
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# 拟合模型
lrModel = lr.fit(training)

# 打印Logistic回归的系数和截距
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# 我们也可以在二分类问题上用多项分布族
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# 拟合模型
mlrModel = mlr.fit(training)

# 打印多项分布族Logistic回归的系数和截距
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))
```

`spark.ml`中Logistic回归的实现也支持提取一个训练集上一个模型的摘要。请注意，作为`DataFrame`存储在`LogisticRegressionSummary`中的预测和指标被标注为`@transient`，因此仅对驱动程序可用。

[`LogisticRegressionTrainingSummary`](api/python/pyspark.ml.md#pyspark.ml.classification.LogisticRegressionSummary)
提供[`LogisticRegressionModel`](api/python/pyspark.ml.md#pyspark.ml.classification.LogisticRegressionModel)的一个摘要。
对于二分类的情况, 还有些额外指标，比如ROC曲线。请参考[`BinaryLogisticRegressionTrainingSummary`](api/python/pyspark.ml.md#pyspark.ml.classification.BinaryLogisticRegressionTrainingSummary)。

接着之前的例子:

```python
from pyspark.ml.classification import LogisticRegression

# 从之前的例子中训练返回的LogisticRegressionModel实例中提取摘要
trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC。
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
lr.setThreshold(bestThreshold)
```

### 多项Logistic回归

通过多项logistic(softmax)回归支持多分类问题。在多项Logistic回归中，
算法产生$K$组系数，或一个$K \times J$维矩阵，其中$K$是结果类别数，$J$是特征数。如果算法带一个截距项拟合，则还得到一个长为$K$的截距向量。

  > `coefficientMatrix`为多项分布系数，`interceptVector`为截距。

  > 一个以多项分布族训练的Logistic回归模型不支持`coefficients`和`intercept`方法。可用`coefficientMatrix`和`interceptVector`替代。

结果类别的条件概率$k \in \{1, 2, ..., K\}$由softmax函数刻画。

`\[
   P(Y=k|\mathbf{X}, \boldsymbol{\beta}_k, \beta_{0k}) =  \frac{e^{\boldsymbol{\beta}_k \cdot \mathbf{X}  + \beta_{0k}}}{\sum_{k'=0}^{K-1} e^{\boldsymbol{\beta}_{k'} \cdot \mathbf{X}  + \beta_{0k'}}}
\]`

我们使用一个多项分布响应模型最小化负对数似然，配合elastic-net惩罚以控制过拟合。

`\[
\min_{\beta, \beta_0} -\left[\sum_{i=1}^L w_i \cdot \log P(Y = y_i|\mathbf{x}_i)\right] + \lambda \left[\frac{1}{2}\left(1 - \alpha\right)||\boldsymbol{\beta}||_2^2 + \alpha ||\boldsymbol{\beta}||_1\right]
\]`

详细推导请参考[here](https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-linear_model)。

**样例**

下面的例子展示了如何训练一个带elastic net正则化的多分类Logistic回归模型，并且提取多分类训练的摘要来评估模型。

```python
from pyspark.ml.classification import LogisticRegression

# 载入训练数据
training = spark \
    .read \
    .format("libsvm") \
    .load("data/mllib/sample_multiclass_classification_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# 拟合模型
lrModel = lr.fit(training)

# 打印多项Logistic回归的参数和截距
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, 我们可以检查每个标签上的指标
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
```

## 决策树分类器

决策树以及其集成算法是机器学习分类和回归问题中非常流行的算法。
关于`spark.ml`实现的更多信息可以在[决策树](#决策树)一节中找到。

**样例**

下面的例子导入LibSVM格式数据，并将之划分为训练数据和测试数据。使用第一部分数据进行训练，剩下数据来测试。训练之前我们使用了两种数据预处理方法来对特征进行转换，并且向`DataFrame`添加了元数据来让树结构算法能够识别。

关于参数的更多细节可以在[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.DecisionTreeClassifier)中找到。

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load the data stored in LIBSVM format as a DataFrame.
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)
```

## 随机森林分类器

随机森林是常用的分类和回归算法。
关于`spark.ml`实现的更多信息可以在[随机森林](#随机森林)一节中找到。

**样例**

下面的例子导入LibSVM格式数据，并将之划分为训练数据和测试数据。使用第一部分数据进行训练，剩下数据来测试。训练之前我们使用了两种数据预处理方法来对特征进行转换，并且向`DataFrame`添加了元数据来让树结构算法能够识别。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.RandomForestClassifier)了解更多细节。

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only
```

## 梯度提升树分类器

梯度提升树基于决策树的集成，是一种常用的分类和回归算法。
关于`spark.ml`实现的更多信息可以在[梯度提升树](#梯度提升树)一节中找到。

**样例**

下面的例子导入LibSVM格式数据，并将之划分为训练数据和测试数据。使用第一部分数据进行训练，剩下数据来测试。训练之前我们使用了两种数据预处理方法来对特征进行转换，并且向`DataFrame`添加了元数据来让树结构算法能够识别。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.GBTClassifier)了解更多细节。

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  # summary only
```

## 多层感知机

多层感知机是基于前向人工神经网络[feedforward artificial neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network)的一种分类器。
多层感知机含有多层节点，每层节点与网络的下一层节点完全连接。输入层的节点代表输入数据，其他层的节点通过将输入数据与层上节点的权重`$\wv$`以及偏差`$\bv$`线性组合且应用一个激活函数，得到该层输出。
`$K+1$`层多层感知机分类器可以写成如下矩阵形式：
`\[
\mathrm{y}(\x) = \mathrm{f_K}(...\mathrm{f_2}(\wv_2^T\mathrm{f_1}(\wv_1^T \x+b_1)+b_2)...+b_K)
\]`
中间层节点使用sigmoid(logistic)函数：
`\[
\mathrm{f}(z_i) = \frac{1}{1 + e^{-z_i}}
\]`
输出层使用softmax函数:
`\[
\mathrm{f}(z_i) = \frac{e^{z_i}}{\sum_{k=1}^N e^{z_k}}
\]`
输出层中`$N$`代表类别数目。

多层感知机通过反向传播来学习模型，其中我们使用logistic损失函数以及L-BFGS优化算法。

**样例**

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.MultilayerPerceptronClassifier)了解更多细节。

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 载入训练数据
data = spark.read.format("libsvm")\
    .load("data/mllib/sample_multiclass_classification_data.txt")

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# 训练模型
model = trainer.fit(train)

# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
```

## 线性支持向量机

[支持向量机](https://en.wikipedia.org/wiki/Support_vector_machine)构造一个高维或无限维空间中的一个超平面或超平面的集合，可用于分类、回归或其他任务。
直观来看，一个好的超平面形成的分割需要离任何类别最近的训练数据点距离最大（所谓functional margin），因为一般来说类间距越大分类器泛化误差越低。
Spark ML中的`LinearSVC`用线性SVM支持二分类问题。内部使用OWLQN算法优化[Hinge损失](https://en.wikipedia.org/wiki/Hinge_loss)。

**样例**

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.LinearSVC)了解更多细节。

```python
from pyspark.ml.classification import LinearSVC

# 载入训练数据
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lsvc = LinearSVC(maxIter=10, regParam=0.1)

# 拟合模型
lsvcModel = lsvc.fit(training)

# 打印线性支持向量机的系数和截距
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))
```

## 一对多分类器

[OneVsRest](http://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest)将一个给定的二分类算法有效地扩展到多分类问题中，也叫做“One-vs-All.”。

`OneVsRest`作为一个`Estimator`来实现。它采用一个基础的`Classifier`然后对于k个类别分别创建二分类问题。类别i的二分类分类器用来预测类别为i还是不为i，即将i类和其他类别区分开来。

最后，通过依次对k个二分类分类器进行评估，取置信最高的分类器的标签作为i类别的标签。

**样例**

下面的例子导入[鸢尾花数据集](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale)，解析为`DataFrame`，用`OneVsRest`分类。计算测试集错误来衡量算法准确率。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.OneVsRest)了解更多细节。

```python
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# load data file.
inputData = spark.read.format("libsvm") \
    .load("data/mllib/sample_multiclass_classification_data.txt")

# generate the train/test split.
(train, test) = inputData.randomSplit([0.8, 0.2])

# instantiate the base classifier.
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(classifier=lr)

# train the multiclass model.
ovrModel = ovr.fit(train)

# score the model on test data.
predictions = ovrModel.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# compute the classification error on test data.
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
```

## 朴素Bayes

[朴素Bayes分类器](http://en.wikipedia.org/wiki/Naive_Bayes_classifier)是基于贝叶斯定理与（朴素的）特征条件独立假设的分类方法。

朴素Bayes可以非常高效地训练。通过在训练数据上一次遍历，对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率。预测时在没有其它可用信息下，我们会选择条件概率最大的类别作为此待分类项应属的类别。

MLlib支持[多项朴素Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes)
和[Bernoulli朴素Bayes](http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html)。

*输入数据*:
这类模型常用于[文档分类](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)。
此处每个观测是一个文档，每个词项表示为一个特征。
一个特征值为该词项的频率（多项朴素Bayes）或0/1指示该词项是否出现在文档中（Bernoulli朴素Bayes）。
特征值必须*非负*。模型类型由一个可选参数`"multinomial"`或`"bernoulli"`决定，默认值是`"multinomial"`。
对于文档分类，输入特征向量经常是稀疏向量。
由于训练数据仅使用一次，没必要缓存它。

[附加平滑](http://en.wikipedia.org/wiki/Lidstone_smoothing)可通过参数$\lambda$设定（默认值$1.0$）。

**样例**

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.classification.NaiveBayes)了解更多细节。

```python
from pyspark.ml.分类import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 载入训练数据
data = spark.read.format("libsvm") \
    .load("data/mllib/sample_libsvm_data.txt")

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# 训练模型
model = nb.fit(train)

# select example rows to display。
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
```

# 回归

## 线性回归

线性回归模型与摘要的接口类似于Logistic回归。

  > 当在有非零常数列的数据集上不带截距拟合`LogisticRegressionModel`时，Spark MLlib对非零常数列输出零系数。这种行为与R glmnet相同但是不同于LIBSVM。

**样例**

下面的例子展示了训练一个elastic net正则化的线性回归模型并提取模型摘要统计。
<!--- TODO: Add python model summaries once implemented -->

关于参数的更多细节可以在[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.LinearRegression)中找到。

```python
from pyspark.ml.regression import LinearRegression

# 载入训练数据
training = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# 拟合模型
lrModel = lr.fit(training)

# 打印线性回归系数和截距
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# 训练集上的模型摘要并打印一些指标
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
```

## 广义线性模型

与线性回归假设输出服从高斯分布不同，[广义线性模型](https://en.wikipedia.org/wiki/Generalized_linear_model)（GLMs）指定线性模型的因变量$Y_i$服从[指数族分布](https://en.wikipedia.org/wiki/Exponential_family)。
Spark的`GeneralizedLinearRegression`接口允许指定GLMs包括线性回归、泊松回归、逻辑回归等来处理多种预测问题。
目前`spark.ml`仅支持指数族分布中的一部分类型，[如下](#available-families)：

**注意**：目前Spark的`GeneralizedLinearRegression`仅支持最多4096个特征，如果特征超过4096个将会抛出异常。请看[advanced section](ml-advanced.md)了解更多细节。
对于线性回归和Logistic回归，如果模型特征数量太多，则可通过`LinearRegression`和`LogisticRegression`来训练。

GLMs要求的指数型分布可以为正则或者自然形式[自然形式指数族分布](https://en.wikipedia.org/wiki/Natural_exponential_family)。自然指数族分布为如下形式：

$$
f_Y(y|\theta, \tau) = h(y, \tau)\exp{\left( \frac{\theta \cdot y - A(\theta)}{d(\tau)} \right)}
$$

其中$\theta$是强度参数，$\tau$是分散度参数。在GLM中因变量$Y_i$服从自然指数族分布：

$$
Y_i \sim f\left(\cdot|\theta_i, \tau \right)
$$

其中强度参数$\theta_i$与因变量$\mu_i$的期望值联系如下：

$$
\mu_i = A'(\theta_i)
$$

其中$A'(\theta_i)$由所选择的分布形式所决定。GLMs同样允许指定连接函数，连接函数决定了因变量期望值与 _线性预测器_ $\eta_i$之间的关系：

$$
g(\mu_i) = \eta_i = \vec{x_i}^T \cdot \vec{\beta}
$$

通常，连接函数的选择如$A' = g^{-1}$，在强度参数$\theta$与线性预测器$\eta$之间产生一个简单的关系。这种情况下，连接函数$g(\mu)$也称为正则连接函数：

$$
\theta_i = A'^{-1}(\mu_i) = g(g^{-1}(\eta_i)) = \eta_i
$$

GLM通过最大化似然函数来求得回归系数$\vec{\beta}$：

$$
\max_{\vec{\beta}} \mathcal{L}(\vec{\theta}|\vec{y},X) =
\prod_{i=1}^{N} h(y_i, \tau) \exp{\left(\frac{y_i\theta_i - A(\theta_i)}{d(\tau)}\right)}
$$

其中强度参数$\theta_i$和回归系数$\vec{\beta}$的联系如下：

$$
\theta_i = A'^{-1}(g^{-1}(\vec{x_i} \cdot \vec{\beta}))
$$

Spark的广义线性模型接口也提供摘要统计来诊断GLM模型的拟合程度，包括残差、p值、偏度、Akaike信息量等等。

可参考更全面的广义线性模型和应用[复习](http://data.princeton.edu/wws509/notes/)。

###  可用的分布族

<table class="table">
  <thead>
    <tr>
      <th>分布族</th>
      <th>因变量类型</th>
      <th>支持的连接类型</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>高斯</td>
      <td>连续型</td>
      <td>Identity*, Log, Inverse</td>
    </tr>
    <tr>
      <td>二项</td>
      <td>二值型</td>
      <td>Logit*, Probit, CLogLog</td>
    </tr>
    <tr>
      <td>泊松</td>
      <td>计数型</td>
      <td>Log*, Identity, Sqrt</td>
    </tr>
    <tr>
      <td>伽马</td>
      <td>连续型</td>
      <td>Inverse*, Idenity, Log</td>
    </tr>
    <tr>
      <td>[Tweedie](https://en.wikipedia.org/wiki/Tweedie_distribution)</td>
      <td>零膨胀连续型</td>
      <td>幂连接函数</td>
    </tr>
    <tfoot><tr><td colspan="4">* 正则连接函数</td></tr></tfoot>
  </tbody>
</table>

**样例**

下面的例子展示了训练一个一个高斯响应与恒等连接函数的GLM并提取模型摘要统计。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.GeneralizedLinearRegression)了解更多细节。

```python
from pyspark.ml.regression import GeneralizedLinearRegression

# 载入训练数据
dataset = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")

glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

# 拟合模型
model = glr.fit(dataset)

# 打印广义线性模型的系数和截距
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# 训练集上的模型摘要并打印一些指标
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()
```

## 回归树

决策树以及其集成算法是机器学习分类和回归问题中非常流行的算法。
关于`spark.ml`实现的更多信息可以在[决策树](#决策树)一节中找到。

**样例**

下面的例子导入LibSVM格式数据，并将之划分为训练数据和测试数据。使用第一部分数据进行训练，剩下数据来测试。训练之前我们使用了一种数据预处理方法来对特征进行转换，并且向`DataFrame`添加了元数据来让树结构算法能够识别。

关于参数的更多细节可以在[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.DecisionTreeRegressor)中找到。

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load the data stored in LIBSVM format as a DataFrame.
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
# summary only
print(treeModel)
```

## 随机森林回归

随机森林是机器学习分类和回归问题中非常流行的算法。
关于`spark.ml`实现的更多信息可以在[随机森林](#随机森林)一节中找到。

**样例**

下面的例子导入LibSVM格式数据，并将之划分为训练数据和测试数据。使用第一部分数据进行训练，剩下数据来测试。训练之前我们使用了一种数据预处理方法来对特征进行转换，并且向`DataFrame`添加了元数据来让树结构算法能够识别。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.RandomForestRegressor)了解更多细节。

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = model.stages[1]
print(rfModel)  # summary only
```

## 梯度提升树回归

梯度提升树回归是机器学习分类和回归问题中非常流行的算法。
关于`spark.ml`实现的更多信息可以在[梯度提升树](#梯度提升树)一节中找到。

**样例**

**注意**：下面的例子中，`GBTRegressor`仅迭代了一次，在实际操作中是不现实的。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.GBTRegressor)了解更多细节。

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

gbtModel = model.stages[1]
print(gbtModel)  # summary only
```

## 生存回归


在`spark.ml`中，我们实现[加速失效时间模型（AFT）](https://en.wikipedia.org/wiki/Accelerated_failure_time_model)，对于截尾数据它是一个参数化生存回归的模型。
它描述了一个有对数生存时间的模型，所以它也常被称为生存分析的对数线性模型。
不同于应对同样问题的[比例危险模型](https://en.wikipedia.org/wiki/Proportional_hazards_model)，AFT模型中每个实例对目标函数的贡献是独立的，所以其更容易并行化。

给定协变量$x^{'}$的值，对于$i = 1, ..., n$，可能右截尾的随机生存时间$t_{i}$，AFT模型下的似然函数如下：

`\[
L(\beta,\sigma)=\prod_{i=1}^n[\frac{1}{\sigma}f_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})]^{\delta_{i}}S_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})^{1-\delta_{i}}
\]`

其中$\delta_{i}$是指示事件i发生了，即有无检测到。
令$\epsilon_{i}=\frac{\log{t_{i}}-x^{'}\beta}{\sigma}$，则对数似然函数为以下形式：

`\[
\iota(\beta,\sigma)=\sum_{i=1}^{n}[-\delta_{i}\log\sigma+\delta_{i}\log{f_{0}}(\epsilon_{i})+(1-\delta_{i})\log{S_{0}(\epsilon_{i})}]
\]`

其中$S_{0}(\epsilon_{i})$是基线生存函数，$f_{0}(\epsilon_{i})$是对应的密度函数。

最常用的AFT模型基于Weibull分布的生存时间。
Weibull分布的生存时间对应于生存时间对数的极值分布，$S_{0}(\epsilon)$函数如下：

`\[
S_{0}(\epsilon_{i})=\exp(-e^{\epsilon_{i}})
\]`

$f_{0}(\epsilon_{i})$函数如下：

`\[
f_{0}(\epsilon_{i})=e^{\epsilon_{i}}\exp(-e^{\epsilon_{i}})
\]`

Weibull分布的生存时间AFT模型对数似然函数如下：

`\[
\iota(\beta,\sigma)= -\sum_{i=1}^n[\delta_{i}\log\sigma-\delta_{i}\epsilon_{i}+e^{\epsilon_{i}}]
\]`

由于最小化负对数似然函数等于最大化后验概率，所以我们要优化的损失函数为$-\iota(\beta,\sigma)$。
分别对$\beta$以及$\log\sigma$求导：

`\[
\frac{\partial (-\iota)}{\partial \beta}=\sum_{1=1}^{n}[\delta_{i}-e^{\epsilon_{i}}]\frac{x_{i}}{\sigma}
\]`
`\[
\frac{\partial (-\iota)}{\partial (\log\sigma)}=\sum_{i=1}^{n}[\delta_{i}+(\delta_{i}-e^{\epsilon_{i}})\epsilon_{i}]
\]`

可以证明AFT模型是一个凸优化问题，即是说找到凸函数$-\iota(\beta,\sigma)$的最小值取决于系数向量$\beta$以及尺度参数的对数$\log\sigma$。
在`spark.ml`中实现的优化算法为L-BFGS。
这个实现与R中的生存函数相匹配
[survreg](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html)

  > 当在有非零常数列的数据集上不带截距拟合`AFTSurvivalRegressionModel`时，Spark MLlib对非零常数列输出零系数。这种行为不同于R的`survival::survreg`。

**样例**

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.AFTSurvivalRegression)了解更多细节。

```python
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.linalg import Vectors

training = spark.createDataFrame([
    (1.218, 1.0, Vectors.dense(1.560, -0.605)),
    (2.949, 0.0, Vectors.dense(0.346, 2.158)),
    (3.627, 0.0, Vectors.dense(1.380, 0.231)),
    (0.273, 1.0, Vectors.dense(0.520, 1.151)),
    (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
quantileProbabilities = [0.3, 0.6]
aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                            quantilesCol="quantiles")

model = aft.fit(training)

# Print the coefficients, intercept and scale parameter for AFT survival regression
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))
print("Scale: " + str(model.scale))
model.transform(training).show(truncate=False)
```

## 保序回归
[保序回归](http://en.wikipedia.org/wiki/Isotonic_regression)是一种回归算法。保序回归的形式化问题是给定一个实数的有限集合`$Y = {y_1, y_2, ..., y_n}$`表示观测到的因变量，`$X = {x_1, x_2, ..., x_n}$`表示未知因变量，拟合模型最小化函数

`\begin{equation}
  f(x) = \sum_{i=1}^n w_i (y_i - x_i)^2
\end{equation}`

使得满足全序`$x_1\le x_2\le ...\le x_n$`，其中`$w_i$`是正权重。
其结果函数称为保序回归，而且其解是唯一的。
它可以被视为有顺序约束下的最小二乘法问题。
实际上最好拟合原始数据点的保序回归是一个[单调函数](http://en.wikipedia.org/wiki/Monotonic_function)。

我们实现了[pool adjacent violators算法](http://doi.org/10.1198/TECH.2010.10111)，它采用一种[并行保序回归](http://doi.org/10.1007/978-3-642-99789-1_10)。
训练数据是一个`DataFrame`，包含标签、特征值以及权重三列。
另外保序算法还有一个参数名为$isotonic$默认为`true`，它指定保序回归为保序（单调递增）或者反序（单调递减）。

训练返回一个保序回归模型，可以被用于来预测已知或者未知特征值的标签。保序回归的结果是分段线性函数，预测规则如下：

* 如果预测输入与训练中的特征值完全匹配，则返回相应标签。如果一个特征值对应多个预测标签值，则返回其中一个，具体是哪一个未指定（类似`java.util.Arrays.binarySearch`）。

* 如果预测输入比训练中的所有特征值都高（或者都低），则相应返回最高特征值或者最低特征值对应标签。如果一个特征值对应多个预测标签值，则相应返回其中最高值或者最低值。

* 如果预测输入落入两个特征值之间，则预测将会是一个分段线性函数，其值由两个最近的特征值的预测值插值计算得到。如果一个特征值对应多个预测标签值，则使用上述两种情况中的处理方式解决。

**样例**

请参考[`IsotonicRegression`的Python文档](api/python/pyspark.ml.md#pyspark.ml.regression.IsotonicRegression)了解API的更多细节。

```python
from pyspark.ml.regression import IsotonicRegression

# Loads data.
dataset = spark.read.format("libsvm")\
    .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")

# Trains an isotonic regression model.
model = IsotonicRegression().fit(dataset)
print("Boundaries in increasing order: %s\n" % str(model.boundaries))
print("Predictions associated with the boundaries: %s\n" % str(model.predictions))

# Makes predictions.
model.transform(dataset).show()
```

# 线性方法

我们实现了常用的线性方法比如logistic回归与带$L_1$或$L_2$正则化的线性最小方。
请参考[RDD接口的线性方法指南](mllib-linear-methods.md)了解实现与调优的更多细节，这些信息仍然有效。

我们也包含一个[Elastic net](http://en.wikipedia.org/wiki/Elastic_net_regularization)的`DataFrame`，一个[Zou et al, Regularization and variable selection via the elastic net](http://users.stat.umn.edu/~zouxx019/Papers/elasticnet.pdf)中提出的混合$L_1$与$L_2$正则化。
在数学上定义为一个$L_1$与$L_2$正则化项的凸组合：

`\[
\alpha \left( \lambda \|\wv\|_1 \right) + (1-\alpha) \left( \frac{\lambda}{2}\|\wv\|_2^2 \right) , \alpha \in [0, 1], \lambda \geq 0
\]`

通过恰当设置$\alpha$，$L_1$与$L_2$正则化是elastic net的特例。举例来说，如果一个[线性回归](https://en.wikipedia.org/wiki/Linear_regression)模型以elastic net参数$\alpha$设为$1$来训练，那么它等价于[Lasso](http://en.wikipedia.org/wiki/Least_squares#Lasso_method)模型。
反之若$\alpha$设为$0$，训练出的模型成为一个[岭回归](http://en.wikipedia.org/wiki/Tikhonov_regularization)模型。
我们为带elastic net正则化的线性回归和logistic回归两者实现了管道API。

# 决策树

[决策树](http://en.wikipedia.org/wiki/Decision_tree_learning)以及其集成算法是机器学习分类和回归问题中非常流行的算法。

决策树因其易解释性、可处理类别特征、易扩展到多分类问题、不需特征缩放以及能够捕捉非线性和特征相互作用等性质被广泛使用。树集成算法如随机森林以及boosting算法几乎是解决分类和回归问题中表现最优的算法。

决策树是一个贪心算法递归地将特征空间划分为两个部分，在同一个叶子节点的数据最后会拥有同样的标签。每次划分通过贪心的以获得最大信息增益为目的，从可选择的分裂方式中选择最佳的分裂节点。节点不纯度有节点所含类别的同质性来衡量。工具提供为分类提供两种不纯度衡量（基尼不纯度和熵），为回归提供一种不纯度衡量（方差）。

`spark.ml`支持二分类、多分类以及回归的决策树算法，适用于连续特征以及类别特征。这里的实现以行划分数据，容许百万至十亿级实例的分布式训练。

用户可以在[MLlib决策树指南](mllib-decision-tree.md)中找到决策树算法的更多信息。
这个API与[之前的MLlib决策树API](mllib-decision-tree.md)之间的主要差别如下:

* 支持机器学习管道
* 区分了用于分类或回归的决策树
* 使用`DataFrame`元数据来区分连续和类别特征

决策树的管道API比之前的API提供了稍微多一点的功能。对于分类问题，工具可以返回属于每种类别的概率（类别条件概率），对于回归问题工具可以返回预测在偏置样本上的方差。

树集成算法（随机森林和梯度提升树）在[树集成](#树集成)一节中说明。

## 输入和输出

我们在这里列出了输入和输出（预测）列的类型。
所有输出列都是可选的；若需要排除一个列，将它的对应参数设为一个空字符串。

### 输入列

<table class="table">
  <thead>
    <tr>
      <th align="left">参数名</th>
      <th align="left">类型</th>
      <th align="left">默认值</th>
      <th align="left">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>labelCol</td>
      <td>Double</td>
      <td>"label"</td>
      <td>预测标签</td>
    </tr>
    <tr>
      <td>featuresCol</td>
      <td>Vector</td>
      <td>"features"</td>
      <td>特征向量</td>
    </tr>
  </tbody>
</table>

### 输出列

<table class="table">
  <thead>
    <tr>
      <th align="left">参数名</th>
      <th align="left">类型</th>
      <th align="left">默认值</th>
      <th align="left">描述</th>
      <th align="left">备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>predictionCol</td>
      <td>Double</td>
      <td>"prediction"</td>
      <td>预测标签</td>
      <td></td>
    </tr>
    <tr>
      <td>rawPredictionCol</td>
      <td>Vector</td>
      <td>"rawPrediction"</td>
      <td>类别数长度的向量，表示在做出预测的树节点上的训练实例计数</td>
      <td>仅限分类</td>
    </tr>
    <tr>
      <td>probabilityCol</td>
      <td>Vector</td>
      <td>"probability"</td>
      <td>类别数长度的向量，等于一个多项分布归一化的`rawPrediction`</td>
      <td>仅限分类</td>
    </tr>
    <tr>
      <td>varianceCol</td>
      <td>Double</td>
      <td></td>
      <td>有偏差的预测样本方差</td>
      <td>仅限回归</td>
    </tr>
  </tbody>
</table>


# 树集成

`DataFrame`API支持两种主要的树集成算法：[随机森林](http://en.wikipedia.org/wiki/Random_forest)和[梯度提升树](http://en.wikipedia.org/wiki/Gradient_boosting)。
两者都使用[`spark.ml`决策树](#决策树)作为基础模型。

用户可以在[MLlib集成指南](mllib-ensembles.md)中找到集成算法的更多信息。
在这一节中，我们演示集成的`DataFrame`API。

这个API与[之前的MLlib集成API](mllib-ensembles.md)之间的主要差别如下:

* 支持`DataFrame`和机器学习管道
* 区分了分类或回归
* 使用`DataFrame`元数据来区分连续和类别特征
* 随机森林功能更多：估计特征重要性，还有对于分类问题，可以返回属于每种类别的概率（类别条件概率）。

## 随机森林

[随机森林](http://en.wikipedia.org/wiki/Random_forest)是[决策树](#决策树)的集成算法。

随机森林包含多个决策树来降低过拟合的风险。随机森林同样具有易解释性、可处理类别特征、易扩展到多分类问题、不需特征缩放等性质。

随机森林分别训练一系列的决策树，所以训练过程是并行的。因算法中加入随机过程，所以每个决策树又有少量区别。通过合并每个树的预测结果来减少预测的方差，提高在测试集上的性能表现。

随机性体现：

* 每次迭代时，对原始数据进行二次抽样来获得不同的训练数据。

* 对于每个树节点，考虑不同的随机特征子集来进行分裂。

除此之外，决策时的训练过程和单独决策树训练过程相同。

对新实例进行预测时，随机森林需要整合其各个决策树的预测结果。回归和分类问题的整合的方式略有不同。分类问题采取投票制，每个决策树投票给一个类别，获得最多投票的类别为最终结果。回归问题每个树得到的预测结果为实数，最终的预测结果为各个树预测结果的平均值。

`spark.ml`支持二分类、多分类以及回归的随机森林算法，同时适用于连续特征和类别特征。

请参考[`spark.mllib`的随机森林文档](mllib-ensembles.md#random-forests)了解算法的更多信息。

## 输入和输出

我们在这里列出了输入和输出（预测）列的类型。
所有输出列都是可选的；若需要排除一个列，将它的对应参数设为一个空字符串。

#### 输入列

<table class="table">
  <thead>
    <tr>
      <th align="left">参数名</th>
      <th align="left">类型</th>
      <th align="left">默认值</th>
      <th align="left">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>labelCol</td>
      <td>Double</td>
      <td>"label"</td>
      <td>预测标签</td>
    </tr>
    <tr>
      <td>featuresCol</td>
      <td>Vector</td>
      <td>"features"</td>
      <td>特征向量</td>
    </tr>
  </tbody>
</table>

### 输出（预测）列

<table class="table">
  <thead>
    <tr>
      <th align="left">参数名</th>
      <th align="left">类型</th>
      <th align="left">默认值</th>
      <th align="left">描述</th>
      <th align="left">备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>predictionCol</td>
      <td>Double</td>
      <td>"prediction"</td>
      <td>预测标签</td>
      <td></td>
    </tr>
    <tr>
      <td>rawPredictionCol</td>
      <td>Vector</td>
      <td>"rawPrediction"</td>
      <td>类别数长度的向量，表示在做出预测的树节点上的训练实例计数</td>
      <td>仅限分类</td>
    </tr>
    <tr>
      <td>probabilityCol</td>
      <td>Vector</td>
      <td>"probability"</td>
      <td>类别数长度的向量，等于一个多项分布归一化的`rawPrediction`</td>
      <td>仅限分类</td>
    </tr>
  </tbody>
</table>


## 梯度提升树

[梯度提升树（GBT）](http://en.wikipedia.org/wiki/Gradient_boosting)是一种决策树的集成算法。它通过反复迭代训练决策树来最小化损失函数。决策树类似，梯度提升树具有可处理类别特征、易扩展到多分类问题、不需特征缩放等性质。`Spark.ml`通过使用现有决策树工具来实现。

梯度提升树依次迭代训练一系列的决策树。在一次迭代中，算法使用现有的集成来对每个训练实例的类别进行预测，然后将预测结果与真实的标签值进行比较。通过重新标记，来赋予预测结果不好的实例更高的权重。所以，在下次迭代中，决策树会对先前的错误进行修正。

对实例标签进行重新标记的机制由损失函数来指定。每次迭代过程中，梯度迭代树在训练数据上进一步减少损失函数的值。`Spark.ml`为分类问题提供一种损失函数（Log Loss），为回归问题提供两种损失函数（平方误差与绝对误差）。

`Spark.ml`支持二分类的梯度提升树算法，适用于连续特征以及类别特征。

请参考[`spark.mllib`梯度提升树文档](mllib-ensembles.md#gradient-boosted-trees-gbts)了解算法的更多信息。


## 输入和输出

我们在这里列出了输入和输出（预测）列的类型。
所有输出列都是可选的；若需要排除一个列，将它的对应参数设为一个空字符串。

#### 输入列

<table class="table">
  <thead>
    <tr>
      <th align="left">参数名</th>
      <th align="left">类型</th>
      <th align="left">默认值</th>
      <th align="left">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>labelCol</td>
      <td>Double</td>
      <td>"label"</td>
      <td>预测标签</td>
    </tr>
    <tr>
      <td>featuresCol</td>
      <td>Vector</td>
      <td>"features"</td>
      <td>特征向量</td>
    </tr>
  </tbody>
</table>

**注意**：`GBTClassifier`目前仅支持二值标签。

### 输出（预测）列

<table class="table">
  <thead>
    <tr>
      <th align="left">参数名</th>
      <th align="left">类型</th>
      <th align="left">默认值</th>
      <th align="left">描述</th>
      <th align="left">备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>predictionCol</td>
      <td>Double</td>
      <td>"prediction"</td>
      <td>预测标签</td>
      <td></td>
    </tr>
  </tbody>
</table>

将来`GBTClassifier`也会像`RandomForestClassifier`一样输出`rawPrediction`和`probability`列。
