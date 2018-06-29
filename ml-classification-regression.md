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

  > 当在有非零常数列的数据集上不带截距拟合LogisticRegressionModel时，Spark MLlib对非零常数列输出零系数。这种行为与R glmnet相同但是不同于LIBSVM。

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

决策树以及其集成算法是机器学习分类和回归问题中非常流行的算法。因其易解释性、可处理类别特征、易扩展到多分类问题、不需特征缩放等性质被广泛使用。树集成算法如随机森林以及boosting算法几乎是解决分类和回归问题中表现最优的算法。

决策树是一个贪心算法递归地将特征空间划分为两个部分，在同一个叶子节点的数据最后会拥有同样的标签。每次划分通过贪心的以获得最大信息增益为目的，从可选择的分裂方式中选择最佳的分裂节点。节点不纯度有节点所含类别的同质性来衡量。工具提供为分类提供两种不纯度衡量（基尼不纯度和熵），为回归提供一种不纯度衡量（方差）。

`spark.ml`支持二分类、多分类以及回归的决策树算法，适用于连续特征以及类别特征。另外，对于分类问题，工具可以返回属于每种类别的概率（类别条件概率），对于回归问题工具可以返回预测在偏置样本上的方差。

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

随机森林是决策树的集成算法。随机森林包含多个决策树来降低过拟合的风险。随机森林同样具有易解释性、可处理类别特征、易扩展到多分类问题、不需特征缩放等性质。

随机森林分别训练一系列的决策树，所以训练过程是并行的。因算法中加入随机过程，所以每个决策树又有少量区别。通过合并每个树的预测结果来减少预测的方差，提高在测试集上的性能表现。

随机性体现：

* 每次迭代时，对原始数据进行二次抽样来获得不同的训练数据。

* 对于每个树节点，考虑不同的随机特征子集来进行分裂。

除此之外，决策时的训练过程和单独决策树训练过程相同。

对新实例进行预测时，随机森林需要整合其各个决策树的预测结果。回归和分类问题的整合的方式略有不同。分类问题采取投票制，每个决策树投票给一个类别，获得最多投票的类别为最终结果。回归问题每个树得到的预测结果为实数，最终的预测结果为各个树预测结果的平均值。

`spark.ml`支持二分类、多分类以及回归的随机森林算法，适用于连续特征以及类别特征。

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

梯度提升树是一种决策树的集成算法。它通过反复迭代训练决策树来最小化损失函数。决策树类似，梯度提升树具有可处理类别特征、易扩展到多分类问题、不需特征缩放等性质。`Spark.ml`通过使用现有决策树工具来实现。

梯度提升树依次迭代训练一系列的决策树。在一次迭代中，算法使用现有的集成来对每个训练实例的类别进行预测，然后将预测结果与真实的标签值进行比较。通过重新标记，来赋予预测结果不好的实例更高的权重。所以，在下次迭代中，决策树会对先前的错误进行修正。

对实例标签进行重新标记的机制由损失函数来指定。每次迭代过程中，梯度迭代树在训练数据上进一步减少损失函数的值。`Spark.ml`为分类问题提供一种损失函数（Log Loss），为回归问题提供两种损失函数（平方误差与绝对误差）。

`Spark.ml`支持二分类以及回归的随机森林算法，适用于连续特征以及类别特征。

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

# Regression

## Linear regression

The interface for working with linear regression models and model
summaries is similar to the Logistic回归 case。

  > 当拟合LinearRegressionModel without 截距on dataset with 常数非零column by "l-bfgs" solver, Spark MLlib outputs zero 参数for 常数非零columns。这种行为 is the same as R glmnet 但是不同于LIBSVM。

**样例**

The following
example demonstrates training一个elastic net regularized linear
regression model and 提取模型摘要统计。
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

# 打印参数and 截距for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize 模型 over 训练集 and print out some 指标
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
```

## Generalized linear regression

Contrasted with linear regression where the output is assumed to follow一个Gaussian
distribution, [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs) are specifications of linear models where the response variable $Y_i$ follows some
distribution from the [exponential family of distributions](https://en.wikipedia.org/wiki/Exponential_family)。
Spark's `GeneralizedLinearRegression` interface
allows for flexible specification of GLMs which 可用于various types of
prediction problems including linear regression, Poisson regression, Logistic回归, and others。
Currently in `spark.ml`, only一个subset of the exponential family distributions are supported and they are listed
[below](#available-families)。

**NOTE**: Spark currently only 支持 up to 4096 features through its `GeneralizedLinearRegression`
interface, and will throw一个exception if this constraint is exceeded。See the [advanced section](ml-advanced)了解更多细节。
 Still, for linear and Logistic回归, models with一个increased number of features can be trained
 using the `LinearRegression` and `LogisticRegression` estimators。

GLMs require exponential family distributions that can be written in their "canonical" or "natural" form, aka
[natural exponential family distributions](https://en.wikipedia.org/wiki/Natural_exponential_family)。The form of一个natural exponential family distribution is given as:

$$
f_Y(y|\theta, \tau) = h(y, \tau)\exp{\left( \frac{\theta \cdot y - A(\theta)}{d(\tau)} \right)}
$$

where $\theta$ is the parameter of interest and $\tau$ is一个dispersion parameter。In一个GLM the response variable $Y_i$ is assumed to be drawn from一个natural exponential family distribution:

$$
Y_i \sim f\left(\cdot|\theta_i, \tau \right)
$$

where the parameter of interest $\theta_i$ is related to the expected value of the response variable $\mu_i$ by

$$
\mu_i = A'(\theta_i)
$$

Here, $A'(\theta_i)$ is defined by the form of the distribution selected。GLMs也allow specification
of一个连接函数, which defines the relationship between the expected value of the response variable $\mu_i$
and the so called _linear predictor_ $\eta_i$:

$$
g(\mu_i) = \eta_i = \vec{x_i}^T \cdot \vec{\beta}
$$

Often, the 连接函数 is chosen such that $A' = g^{-1}$, which yields一个simplified relationship
between the parameter of interest $\theta$ and the linear predictor $\eta$。In this case, the link
function $g(\mu)$ is said to be the "canonical" 连接函数。

$$
\theta_i = A'^{-1}(\mu_i) = g(g^{-1}(\eta_i)) = \eta_i
$$

A GLM finds the regression 参数$\vec{\beta}$ which maximize the 似然 function。

$$
\max_{\vec{\beta}} \mathcal{L}(\vec{\theta}|\vec{y},X) =
\prod_{i=1}^{N} h(y_i, \tau) \exp{\left(\frac{y_i\theta_i - A(\theta_i)}{d(\tau)}\right)}
$$

where the parameter of interest $\theta_i$ is related to the regression 参数$\vec{\beta}$
by

$$
\theta_i = A'^{-1}(g^{-1}(\vec{x_i} \cdot \vec{\beta}))
$$

Spark's generalized linear regression interface也提供 summary统计 for diagnosing the
fit of GLM models, including residuals, p-values, deviances, the Akaike information criterion, and
others。

[See here](http://data.princeton.edu/wws509/notes/) for一个more comprehensive review of GLMs and their applications。

###  Available families

<table class="table">
  <thead>
    <tr>
      <th>Family</th>
      <th>Response Type</th>
      <th>Supported Links</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Gaussian</td>
      <td>Continuous</td>
      <td>Identity*, Log, Inverse</td>
    </tr>
    <tr>
      <td>Binomial</td>
      <td>Binary</td>
      <td>Logit*, Probit, CLogLog</td>
    </tr>
    <tr>
      <td>Poisson</td>
      <td>Count</td>
      <td>Log*, Identity, Sqrt</td>
    </tr>
    <tr>
      <td>Gamma</td>
      <td>Continuous</td>
      <td>Inverse*, Idenity, Log</td>
    </tr>
    <tr>
      <td>Tweedie</td>
      <td>Zero-inflated continuous</td>
      <td>Power 连接函数</td>
    </tr>
    <tfoot><tr><td colspan="4">* Canonical Link</td></tr></tfoot>
  </tbody>
</table>

**样例**

下面的例子 demonstrates training一个GLM with一个Gaussian response and identity 连接函数 and 提取模型摘要统计。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.GeneralizedLinearRegression)了解更多细节。

```python
from pyspark.ml.regression import GeneralizedLinearRegression

# 载入训练数据
dataset = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")

glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

# 拟合模型
model = glr.fit(dataset)

# 打印系数and 截距for generalized linear regression model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Summarize 模型 over 训练集 and print out some 指标
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

## Decision tree regression

Decision trees are一个popular family of 分类and regression methods。
More information about the `spark.ml` implementation can be found further in the [section on decision trees](#decision-trees)。

**样例**

下面的例子s load一个dataset in LibSVM format, split it into training and test sets, 训练on the first dataset, and then evaluate on the held-out test set。
We use一个feature transformer to index categorical features, adding metadata to the `DataFrame` which the Decision Tree algorithm can recognize。

关于参数的更多细节可以在[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.DecisionTreeRegressor)中找到。

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load the data stored in LIBSVM format as一个DataFrame。
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them。
# We specify maxCategories so features with > 4 distinct values are treated as continuous。
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train一个DecisionTree model。
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in一个Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# 训练model。 This也runs the indexer。
model = pipeline.fit(trainingData)

# Make predictions。
predictions = model.transform(testData)

# Select example rows to display。
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

## Random forest regression

Random forests are一个popular family of 分类and regression methods。
More information about the `spark.ml` implementation can be found further in the [section on random forests](#random-forests)。

**样例**

下面的例子s load一个dataset in LibSVM format, split it into training and test sets, 训练on the first dataset, and then evaluate on the held-out test set。
We use一个feature transformer to index categorical features, adding metadata to the `DataFrame` which the tree-based 算法 can recognize。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.RandomForestRegressor)了解更多细节。

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load and parse the data file, converting it to一个DataFrame。
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them。
# Set maxCategories so features with > 4 distinct values are treated as continuous。
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train一个RandomForest model。
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# Chain indexer and forest in一个Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# 训练model。 This也runs the indexer。
model = pipeline.fit(trainingData)

# Make predictions。
predictions = model.transform(testData)

# Select example rows to display。
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = model.stages[1]
print(rfModel)  # summary only
```

## Gradient-boosted tree regression

Gradient-boosted trees (GBTs) are一个popular regression method using ensembles of decision trees。
More information about the `spark.ml` implementation can be found further in the [section on GBTs](#gradient-boosted-trees-gbts)。

**样例**

Note: For this example dataset, `GBTRegressor` actually only needs 1 iteration, 但是that will not
be true in general。

请参考[Python API文档](api/python/pyspark.ml.md#pyspark.ml.regression.GBTRegressor)了解更多细节。

```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Load and parse the data file, converting it to一个DataFrame。
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them。
# Set maxCategories so features with > 4 distinct values are treated as continuous。
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train一个GBT model。
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

# Chain indexer and GBT in一个Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# 训练model。 This也runs the indexer。
model = pipeline.fit(trainingData)

# Make predictions。
predictions = model.transform(testData)

# Select example rows to display。
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

gbtModel = model.stages[1]
print(gbtModel)  # summary only
```

## Survival regression


In `spark.ml`, we implement the [Accelerated failure time (AFT)](https://en.wikipedia.org/wiki/Accelerated_failure_time_model)
model which is一个parametric survival regression model for censored data。
It describes一个model for the log of survival time, so it's often called a
log-linear model for survival analysis。不同于a
[Proportional hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) model
designed for the same purpose, the AFT model is easier to parallelize
because each instance contributes to the objective function independently。

Given the values of the covariates $x^{'}$, for random lifetime $t_{i}$ of
subjects i = 1, ..., n, with possible right-censoring,
the 似然 function under the AFT model is given as:
`\[
L(\beta,\sigma)=\prod_{i=1}^n[\frac{1}{\sigma}f_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})]^{\delta_{i}}S_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})^{1-\delta_{i}}
\]`
Where $\delta_{i}$ is the indicator of the event has occurred i.e。uncensored or not。
Using $\epsilon_{i}=\frac{\log{t_{i}}-x^{'}\beta}{\sigma}$, the log-似然 function
assumes the form:
`\[
\iota(\beta,\sigma)=\sum_{i=1}^{n}[-\delta_{i}\log\sigma+\delta_{i}\log{f_{0}}(\epsilon_{i})+(1-\delta_{i})\log{S_{0}(\epsilon_{i})}]
\]`
Where $S_{0}(\epsilon_{i})$ is the baseline survivor function,
and $f_{0}(\epsilon_{i})$ is the corresponding density function。

The most commonly used AFT model is based on the Weibull distribution of the survival time。
The Weibull distribution for lifetime 对应于the extreme value distribution for the
log of the lifetime, and the $S_{0}(\epsilon)$ function is:
`\[
S_{0}(\epsilon_{i})=\exp(-e^{\epsilon_{i}})
\]`
the $f_{0}(\epsilon_{i})$ function is:
`\[
f_{0}(\epsilon_{i})=e^{\epsilon_{i}}\exp(-e^{\epsilon_{i}})
\]`
The log-似然 function for AFT model with一个Weibull distribution of lifetime is:
`\[
\iota(\beta,\sigma)= -\sum_{i=1}^n[\delta_{i}\log\sigma-\delta_{i}\epsilon_{i}+e^{\epsilon_{i}}]
\]`
Due to minimizing the negative log-似然 equivalent to maximum一个posteriori probability,
the loss function we use to optimize is $-\iota(\beta,\sigma)$。
The gradient functions for $\beta$ and $\log\sigma$ respectively are:
`\[
\frac{\partial (-\iota)}{\partial \beta}=\sum_{1=1}^{n}[\delta_{i}-e^{\epsilon_{i}}]\frac{x_{i}}{\sigma}
\]`
`\[
\frac{\partial (-\iota)}{\partial (\log\sigma)}=\sum_{i=1}^{n}[\delta_{i}+(\delta_{i}-e^{\epsilon_{i}})\epsilon_{i}]
\]`

The AFT model can be formulated as一个convex optimization problem,
i.e。the task of finding一个minimizer of一个convex function $-\iota(\beta,\sigma)$
that depends on the 参数vector $\beta$ and the log of scale parameter $\log\sigma$。
The optimization algorithm underlying the implementation is L-BFGS。
The implementation matches the result from R's survival function
[survreg](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html)

  > 当拟合AFTSurvivalRegressionModel without 截距on dataset with 常数非零column, Spark MLlib outputs zero 参数for 常数非零columns。这种行为 is 不同于R survival::survreg。

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

# 打印coefficients, 截距and scale parameter for AFT survival regression
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))
print("Scale: " + str(model.scale))
model.transform(training).show(truncate=False)
```

## Isotonic regression
[Isotonic regression](http://en.wikipedia.org/wiki/Isotonic_regression)
belongs to the family of regression 算法。Formally isotonic regression is一个problem where
given一个finite set of real numbers `$Y = {y_1, y_2, ..., y_n}$` representing observed responses
and `$X = {x_1, x_2, ..., x_n}$` the unknown response values to be fitted
finding一个function that minimizes

`\begin{equation}
  f(x) = \sum_{i=1}^n w_i (y_i - x_i)^2
\end{equation}`

with respect to complete order subject to
`$x_1\le x_2\le ...\le x_n$` where `$w_i$` are positive weights。
The resulting function is called isotonic regression and it is unique。
It can be viewed as least squares problem under order restriction。
Essentially isotonic regression is a
[monotonic function](http://en.wikipedia.org/wiki/Monotonic_function)
best 拟合the original data points。

We implement a
[pool adjacent violators algorithm](http://doi.org/10.1198/TECH.2010.10111)
which uses一个approach to
[parallelizing isotonic regression](http://doi.org/10.1007/978-3-642-99789-1_10)。
The training input is一个DataFrame which contains three columns
label, features and weight。Additionally, IsotonicRegression algorithm has one
optional parameter called $isotonic$ defaulting to true。
This argument specifies if the isotonic regression is
isotonic (monotonically increasing) or antitonic (monotonically decreasing)。

Training returns一个IsotonicRegressionModel that 可用于predict
labels for both known and unknown features。The result of isotonic regression
is treated as piecewise linear function。The rules for prediction therefore are:

* If the prediction input exactly matches一个training feature
  then associated prediction is returned。In case there are multiple predictions with the same
  feature then one of them is returned。Which one is undefined
  (same as java.util.Arrays.binarySearch)。
* If the prediction input is lower or higher than all training features
  then prediction with lowest or highest feature is returned respectively。
  In case there are multiple predictions with the same feature
  then the lowest or highest is returned respectively。
* If the prediction input falls between two training features then prediction is treated
  as piecewise linear function and interpolated value is calculated from the
  predictions of the two closest features。In case there are multiple values
  with the same feature then the same rules as in previous point are used。

**样例**

请参考[`IsotonicRegression` Python docs](api/python/pyspark.ml.md#pyspark.ml.regression.IsotonicRegression) for 更多细节on the API。

```python
from pyspark.ml.regression import IsotonicRegression

# Loads data。
dataset = spark.read.format("libsvm")\
    .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")

# Trains一个isotonic regression model。
model = IsotonicRegression().fit(dataset)
print("Boundaries in increasing order: %s\n" % str(model.boundaries))
print("Predictions associated with the boundaries: %s\n" % str(model.predictions))

# Makes predictions。
model.transform(dataset).show()
```

# Linear methods

We implement popular linear methods such as logistic
regression and linear least squares with $L_1$ or $L_2$正则化。
Refer to [the linear methods guide for the RDD-based API](mllib-linear-methods.md) for
details about implementation and tuning; this information is still relevant。

We也include一个DataFrame API for [Elastic
net](http://en.wikipedia.org/wiki/Elastic_net_regularization),一个hybrid
of $L_1$ and $L_2$正则化 proposed in [Zou et al,正则化
and variable selection via the elastic
net](http://users.stat.umn.edu/~zouxx019/Papers/elasticnet.pdf)。
Mathematically, it is defined as一个convex combination of the $L_1$ and
the $L_2$正则化 terms:
`\[
\alpha \left( \lambda \|\wv\|_1 \right) + (1-\alpha) \left( \frac{\lambda}{2}\|\wv\|_2^2 \right) , \alpha \in [0, 1], \lambda \geq 0
\]`
By setting $\alpha$ properly, elastic net contains both $L_1$ and $L_2$
regularization as special cases。For example, if一个[linear
regression](https://en.wikipedia.org/wiki/Linear_regression) model is
trained with the elastic net parameter $\alpha$ set to $1$, it is
equivalent to a
[Lasso](http://en.wikipedia.org/wiki/Least_squares#Lasso_method) model。
On the other hand, if $\alpha$ is set to $0$, the trained model reduces
to一个[ridge
regression](http://en.wikipedia.org/wiki/Tikhonov_regularization) model。
We implement Pipelines API for both linear regression and logistic
regression with elastic net正则化。

# Decision trees

[Decision trees](http://en.wikipedia.org/wiki/Decision_tree_learning)
and their ensembles are popular methods for the machine learning tasks of
分类and regression。Decision trees are widely used since they are easy to interpret,
handle categorical features, extend to the multiclass 分类setting, do not require
feature scaling, and are able to capture non-linearities and feature interactions。Tree ensemble
算法 such as random forests and boosting are among the top performers for 分类and
regression tasks。

The `spark.ml` implementation 支持 decision trees for binary and multiclass 分类and for regression,
using both continuous and categorical features。The implementation partitions data by rows,
allowing distributed training with millions or even billions of instances。

Users can find more information about the decision tree algorithm in the [MLlib Decision Tree guide](mllib-decision-tree.md)。
The main differences between this API and the [original MLlib Decision Tree API](mllib-decision-tree.md) are:

* support for ML Pipelines
* separation of Decision Trees for 分类vs。regression
* use of DataFrame metadata to distinguish continuous and categorical features


The Pipelines API for Decision Trees offers一个bit more functionality than the original API。
In particular, for分类, users can get the predicted probability of each class (a.k.a。class conditional probabilities);
for regression, users can get the biased sample variance of prediction。

Ensembles of trees (Random Forests and Gradient-Boosted Trees) are described below in the [Tree ensembles section](#tree-ensembles)。

## Inputs and Outputs

We list the input and output (prediction) column types here。
All output columns are optional; to exclude一个output column, set its corresponding Param to一个空字符串。

### Input Columns

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
      <td>Label to predict</td>
    </tr>
    <tr>
      <td>featuresCol</td>
      <td>Vector</td>
      <td>"features"</td>
      <td>Feature vector</td>
    </tr>
  </tbody>
</table>

### Output Columns

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
      <td>Predicted label</td>
      <td></td>
    </tr>
    <tr>
      <td>rawPredictionCol</td>
      <td>Vector</td>
      <td>"rawPrediction"</td>
      <td>Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction</td>
      <td>分类only</td>
    </tr>
    <tr>
      <td>probabilityCol</td>
      <td>Vector</td>
      <td>"probability"</td>
      <td>Vector of length # classes equal to rawPrediction normalized to一个多项distribution</td>
      <td>分类only</td>
    </tr>
    <tr>
      <td>varianceCol</td>
      <td>Double</td>
      <td></td>
      <td>The biased sample variance of prediction</td>
      <td>Regression only</td>
      </tr>
  </tbody>
</table>


# Tree Ensembles

The DataFrame API 支持 two major tree ensemble 算法: [Random Forests](http://en.wikipedia.org/wiki/Random_forest) and [Gradient-Boosted Trees (GBTs)](http://en.wikipedia.org/wiki/Gradient_boosting)。
Both use [`spark.ml` decision trees](ml-classification-regression.md#decision-trees) as their base models。

Users can find more information about ensemble 算法 in the [MLlib Ensemble guide](mllib-ensembles.md)。
In this section, we demonstrate the DataFrame API for ensembles。

The main differences between this API and the [original MLlib ensembles API](mllib-ensembles.md) are:

* support for DataFrames and ML Pipelines
* separation of 分类vs。regression
* use of DataFrame metadata to distinguish continuous and categorical features
* more functionality for random forests: estimates of feature importance, as well as the predicted probability of each class (a.k.a。class conditional probabilities) for分类。

## Random Forests

[Random forests](http://en.wikipedia.org/wiki/Random_forest)
are ensembles of [decision trees](ml-classification-regression.md#decision-trees)。
Random forests combine many decision trees in order to reduce the risk of overfitting。
The `spark.ml` implementation 支持 random forests for binary and multiclass 分类and for regression,
using both continuous and categorical features。

For more information on the algorithm itself, please see the [`spark.mllib` documentation on random forests](mllib-ensembles.md#random-forests)。

### Inputs and Outputs

We list the input and output (prediction) column types here。
All output columns are optional; to exclude一个output column, set its corresponding Param to一个空字符串。

#### Input Columns

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
      <td>Label to predict</td>
    </tr>
    <tr>
      <td>featuresCol</td>
      <td>Vector</td>
      <td>"features"</td>
      <td>Feature vector</td>
    </tr>
  </tbody>
</table>

#### Output Columns (Predictions)

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
      <td>Predicted label</td>
      <td></td>
    </tr>
    <tr>
      <td>rawPredictionCol</td>
      <td>Vector</td>
      <td>"rawPrediction"</td>
      <td>Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction</td>
      <td>分类only</td>
    </tr>
    <tr>
      <td>probabilityCol</td>
      <td>Vector</td>
      <td>"probability"</td>
      <td>Vector of length # classes equal to rawPrediction normalized to一个多项distribution</td>
      <td>分类only</td>
    </tr>
  </tbody>
</table>



## Gradient-Boosted Trees (GBTs)

[Gradient-Boosted Trees (GBTs)](http://en.wikipedia.org/wiki/Gradient_boosting)
are ensembles of [decision trees](ml-classification-regression.md#decision-trees)。
GBTs iteratively 训练decision trees in order to minimize一个loss function。
The `spark.ml` implementation 支持 GBTs for 二分类and for regression,
using both continuous and categorical features。

For more information on the algorithm itself, please see the [`spark.mllib` documentation on GBTs](mllib-ensembles.md#gradient-boosted-trees-gbts)。

### Inputs and Outputs

We list the input and output (prediction) column types here。
All output columns are optional; to exclude一个output column, set its corresponding Param to一个空字符串。

#### Input Columns

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
      <td>Label to predict</td>
    </tr>
    <tr>
      <td>featuresCol</td>
      <td>Vector</td>
      <td>"features"</td>
      <td>Feature vector</td>
    </tr>
  </tbody>
</table>

Note that `GBTClassifier` currently only 支持 binary labels。

#### Output Columns (Predictions)

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
      <td>Predicted label</td>
      <td></td>
    </tr>
  </tbody>
</table>

In the future, `GBTClassifier` will也output columns for `rawPrediction` and `probability`, just as `RandomForestClassifier` does。
