"""
    该案例数据来源于kaggle，该案例数据是300多位骨科病人的疾病数据。
    该案例的目的是通过KNN算法学习该案例数据对新来的骨科疾病数据进行患病或者非患病的判断
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#数据已经准备完毕
#数据的基本信息，数据的概览
data = pd.read_csv('/Users/xuyadan/Data_Analysis/projects/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
print(data.info())
print(data.head(10))
print(data.tail(10))

#数据的统计
#1.看数据的总体统计情况
print(data.describe())
#2.Abnormal和Normal的比较
ax = sns.countplot(x='class',data=data)

#对数据进行清洗，转换
data['label'] = data['class'].map({'Abnormal':1,'Normal':0})
print(data.head(10))
print(data.tail(10))

#分析模型所需的输入数据的格式，对数据进行处理
x = data.iloc[:,:6].values
y = data['label'].values

#对数据进行划分，分为训练数据和测试数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)


#将数据输入到模型中训练
k = [1,3,5,7,9,11,13,15]
model_list = []
acc_list = []
for i in k:
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(x_train,y_train)
    y_pred = KNN.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    model_list.append(KNN)
    acc_list.append(accuracy)
    print('k={}时，准确率为{}'.format(i,accuracy))

#可视化不同k值对准确率的影响，从而找到最佳的k值，然后利用最佳的k值预测新的数据
plt.figure(figsize=(10,8))
plt.plot(acc_list)
#标题
plt.title('The influence on different k values')
#x轴
plt.xlabel('Different K values')
plt.xticks(range(len(k)), k)
#y轴
plt.ylabel('Accuracy Result')
plt.savefig('KNN_values.png')
plt.show()











