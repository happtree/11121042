# 11121042
问题：研究者获取乳房肿块的细针穿刺（FNA），然后生成数字图像。该数据集包含描述图像中细胞核特征的实例。每个实例包括诊断结果：M（恶性）或 B（良性）。我们的任务是在该数据上训练神经网络根据上述特征诊断乳腺癌。

1、下载数据
首先将数据集放置到该机器上，这样我们的 notebook 就可以访问它。你可以使用以下代码：

from google.colab import files
uploaded = files.upload()
结果：

wdbc.data(n/a) - 124103 bytes, last modified: 2019/3/5 - 100% done
 Saving wdbc.data to wdbc.data
另存为breast_cancer.csv:

with open("breast_cancer.csv", 'w') as f:
    f.write(uploaded[uploaded.keys()[0]])
用!ls查看结果如下：

breast_cancer.csv  sample_data    wdbc.data
2、数据预处理
现在数据已经在机器上了，我们使用 pandas 将其输入到项目中。

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('breast_cancer.csv')
查看一下前五行:

dataset.head(5)


 

 现在，分割因变量（Dependent Variables）和自变量（Independent Variables）。

#Seperating dependent and independent variables. 

X = dataset.iloc[:, 2:32].values  #Note: Exclude Last column with all NaN values.
y = dataset.iloc[:, 1].values
Y 包含一列，其中的「M」和「B」分别代表「是」（恶性）和「否」（良性）。我们需要将其编码成数学形式，即「1」和「0」。可以使用 Label Encoder 类别完成该任务。

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)
（如果数据类别多于两类，则使用 OneHotEncoder）

'''#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y).toarray()
y = y[:, 1:]'''
此时查看一下x和y:

X.shape
y.shape
结果：

(568, 30)
(568,)
现在数据已经准备好，我们将其分割成训练集和测试集。在 Scikit-Learn 中使用 train_test_split 可以轻松完成该工作。

#Splitting into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
参数 test_size = 0.2 定义测试集比例。这里，我们将训练集设置为数据集的 80%，测试集占数据集的 20%。

 

3、搭建神经网络
3.1  Keras
Keras 是一种构建人工神经网络的高级 API。它使用 TensorFlow 或 Theano 后端执行内部运行。要安装 Keras，必须首先安装 TensorFlow。CoLaboratory 已经在虚拟机上安装了 TensorFlow。使用以下命令可以检查是否安装 TensorFlow：

!pip show tensorflow
你还可以使用!pip install tensorflow==1.2，安装特定版本的 TensorFlow。

另外，如果你更喜欢用 Theano 后端，可以阅读该文档：https://keras.io/backend/。

安装 Keras：

!pip install -q keras
3.2  然后导入Keras库和包
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
使用 Sequential 和 Dense 类别指定神经网络的节点、连接和规格。如上所示，我们将使用这些自定义网络的参数并进行调整。

3.3  初始化神经网络
为了初始化神经网络，我们将创建一个 Sequential 类的对象。

# Initialising the ANN
classifier = Sequential()
3.4  设计神经网络
对于每个隐藏层，我们需要定义三个基本参数：units、kernel_initializer 和 activation。units 参数定义每层包含的神经元数量。Kernel_initializer 定义神经元在输入数据上运行时的初始权重（详见 https://faroit.github.io/keras-docs/1.2.2/initializations/）。activation 定义数据的激活函数。

输入层和第一个隐藏层：16 个具备统一初始权重的神经元，激活函数为 ReLU。此外，定义参数 input_dim = 30 作为输入层的规格。注意我们的数据集中有 30 个特征列。注：如何确定第一个隐藏层的节点数，对于初学者来说，一种简单方式是：x 和 y 的总和除以 2。如 (30+1)/2 = 15.5 ~ 16，因此，units = 16.
第二层：第二层和第一层一样，不过第二层没有 input_dim 参数.
由于我们的输出是 0 或 1，因此我们可以使用具备统一初始权重的单个单元。但是，这里我们使用 sigmoid 激活函数。
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

# Adding the second hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
3.5  编译
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
3.6  拟合
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
运行人工神经网络，发生反向传播。你将在 CoLaboratory 上看到所有处理过程，而不是在自己的电脑上。

这里 batch_size 是你希望同时处理的输入量。epoch 指数据通过神经网络一次的整个周期。它们在 Colaboratory Notebook 中显示如下：



 

 3.7  进行预测，构建混淆矩阵
训练网络后，就可以在 X_test set 上进行预测，以检查模型在新数据上的性能。在代码单元中输入和执行 cm 查看结果。

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   
混淆矩阵
混淆矩阵是模型做出的正确、错误预测的矩阵表征。该矩阵可供个人调查哪些预测和另一种预测混淆。这是一个 2×2 的混淆矩阵。



 

 混淆矩阵如下所示[cm (Ctrl+Enter)]



上图表示：68 个真负类、0 个假正类、4 个假负类、42 个真正类。很简单。该平方矩阵的大小随着分类类别的增加而增加。

 这个示例中的准确率几乎达到 100%，只有 4 个错误预测。但是并不总是这样。有时你可能需要投入更多时间，研究模型的行为，提出更好、更复杂的解决方案。如果一个网络性能不够好，你需要调整超参数来改进模型。
