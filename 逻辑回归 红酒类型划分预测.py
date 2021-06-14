# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

sns.set(style='whitegrid')


# 读取数据集，以分号为分隔符
data_red = pd.read_csv('winequality-red.csv', sep=';') #红红酒数据
data_white = pd.read_csv('winequality-white.csv', sep=';')#白红酒数据

#查看数据
# data_red.shape
# data_white.shape

# data_red.head()
#data_red.info() #查看数据类型
# data_white.head()
# data_white.info() #查看数据类型
#查看空值
data_white.isnull().any()
data_red.isnull().any()

#进行数据合并
data_red.insert(0, 'type', 'red')
data_white.insert(0, 'type', 'white')


wines = data_red.append(data_white, ignore_index=True)

#查看数据、及其类型
# wines.shape
# wines.info()

# 根据质量等级将记录进行分组
# 低质量为 小于等于 5，  中质量为 {6，7} ，高质量 大于7

wines['quality class'] = wines.quality.apply(lambda q: 'low' if q <= 5 else 'high' if q > 7 else 'medium')

#查看数据、及其类型
# wines.head()
# wines.info()

# wines.apply(lambda c: [c.unique()])
# 查看类型数据
# wines.dtypes.value_counts()
# 生成描述性统计
# wines.describe()

# 保存数据
wines.to_csv('winesdz.csv', index=False)


# 红酒类型的描述性分析统计信息数据


#红红酒
red_wine = round(wines.loc[wines.type == 'red', wines.columns].describe(), 2).T
#白红酒
white_wine = round(wines.loc[wines.type == 'white', wines.columns].describe(), 2).T

pd.concat([red_wine, white_wine], axis=1, keys=['Red Wine', 'White Wine'])


#红酒质量的描述性统计数据

lqs = round(wines.loc[wines['quality class'] == 'low', wines.columns].describe(),2).T
mqs = round(wines.loc[wines['quality class'] == 'medium', wines.columns].describe(),2).T
hqs = round(wines.loc[wines['quality class'] == 'high', wines.columns].describe(),2).T

#合并数据集
pd.concat([lqs, mqs, hqs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 'High Quality Wine'])



#探索性数据分析，各类型的红酒分布
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.suptitle('红酒类型与质量', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

sns.countplot(x='quality',
              data=wines[wines.type == 'red'],
              color='red',
              edgecolor='black',
              ax=ax1)
ax1.set_title('红红酒')
ax1.set_xlabel('质量')
ax1.set_ylabel('频率', size=12)
ax1.set_ylim([0, 2300])


sns.countplot(x='quality',
              data=wines[wines.type == 'white'],
              color='palegreen',
              edgecolor='black',
              ax=ax2)
ax2.set_title("白红酒")
ax2.set_xlabel("质量")
ax2.set_ylabel("频率")
ax2.set_ylim([0, 2300])
plt.show()
f.savefig('ax1.png')
# 以红酒类型和质量等级划分的数据分布。
# 我们可以看到它们是正态分布的。
# 大部分红酒的评级为5至7，而少数红酒的评级为“非常好”{8 – 9}和“非常差”{3 – 4}




# 进行标准化处理
wines = wines.sample(frac=1, random_state=77).reset_index(drop=True)
#通过LabelEncoder()标准化标签，使非数据标签标准化为数字标签
# 创建标签对象
le = LabelEncoder()
#返回编码对象的标签，转为整数，红红酒为0，白红酒为1
y_type = le.fit_transform(wines.type.values)

wines['color'] = y_type



# wines.head()
# wines.info()
# wines.color.unique()

#创建属性数字标签字典
qcl = {'low':0, 'medium': 1, 'high': 2}
y_qclass = wines['quality class'].map(qcl)


#查看红酒类型相关性
wcorr = wines.corr()
#排序
sort_corr_cols = wcorr.color.sort_values(ascending=False).keys()
sort_corr_t = wcorr.loc[sort_corr_cols, sort_corr_cols]

#绘制相关性热力图
plt.figure(figsize=(13.5, 11.5))
sns.heatmap(sort_corr_t,
            annot=True,
            annot_kws=dict(fontsize=14),
            square=True,
            fmt='.2f',
            cmap='coolwarm')
plt.title('以红酒类型划分属性相关性',
          fontsize=14,
          fontweight='bold',
          pad=10)
plt.xticks(rotation=50, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.show()

#红酒类型绘制红酒属性对图
g = sns.pairplot(wines,
                 hue='type',
                 palette={'red' : 'red', 'white' : 'palegreen'},
                 plot_kws=dict(edgecolor='b', linewidth=0.5))

fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)
fig.suptitle('以属性划分红酒类型',
             fontsize=26,
             fontweight='bold')
plt.show()
# 保存图表
g.savefig('pairplot1.png')

#红酒的质量相关性
# sort_corr_cols = wcorr.quality.sort_values(ascending=False).keys()
# sort_corr_q = wcorr.loc[sort_corr_cols,sort_corr_cols]

# 预测酒的类型（红色或白色）

# wines.head()

#特征提取和目标
features = wines.drop(['type', 'quality', 'quality class', 'color'], axis=1).columns
X = wines[features].copy()


# X.head()
y = wines.color.copy()
# y.head()

wines.groupby('color').color.count()

# 红酒类型的数据分布

sns.countplot(x='type',
              data=wines,
              edgecolor='black',
              palette={'red': 'red', 'white': 'palegreen'})

plt.title('红酒类型的数据分布图',
          fontsize=14,
          pad=10)
plt.show()

#逻辑回归，进行预测类型

#将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=77, stratify=y)

# *使用标准缩放器声明建模管道
pipeline = Pipeline([
    ('scl', StandardScaler()),
    ('lr', LogisticRegression(random_state=77))
])


# 调整的超参数
# print(pipeline.get_params())


param_grid = {
    'lr__C': [0.1, 1, 10, 100],
    'lr__tol': [0.001, 0.0001]
}

# Sklearn交叉验证


clf = GridSearchCV(pipeline, param_grid, cv=10)

# 拟合数据并调整模型
clf.fit(X_train, y_train)

#使用cv得出最佳参数
clf.best_params_
clf.best_estimator_

# 模型预测与评估

# 预测新数据
y_pred = clf.predict(X_test)

# 进行模型性能评估
target_names = ['红红酒', '白红酒']
print(classification_report(y_test, y_pred, target_names=target_names), '\n')
print(confusion_matrix(y_test, y_pred))

# 未进行标准化混淆矩阵图热图 与 标准化混淆矩阵图热图

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
f.suptitle('逻辑回归', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

# 未进行标准化混淆矩阵图热图
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat,
            annot=True,
            fmt='d',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax1)

ax1.set_xticklabels(labels=['red', 'white'])
ax1.set_yticklabels(labels=['red', 'white'])
ax1.set_title('未进行标准化混淆矩阵')
ax1.set_xlabel('预测标签')
ax1.set_ylabel('真实标签')

#  标准化混淆矩阵图热图
matn = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(matn,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax2)

ax2.set_xticklabels(labels=['red', 'white'])
ax2.set_yticklabels(labels=['red', 'white'])
ax2.set_title('标准化混淆矩阵')
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')


# 评估模型的准确度
print('模型的预测准确度:')
print(accuracy_score(y_true=y_test, y_pred=y_pred))

# 约为99.4％的准确度




#
#
#
#
# #向量机以红酒质量分类
# #使用标准缩放器声明建模管道
# pipeline = Pipeline([
#     ('scl',StandardScaler()),
#     ('svc',SVC(random_state=77))
# ])
#
#
# # 调整的超参数
# # print(pipeline.get_params())
#
# #调整的超参数
# param_grid = {
#     'svc__C': [0.08, 0.1, 1, 10],
#     'svc__gamma': [5, 1, 0.1, 0.01]
# }
#
# #Sklearn交叉验证
# clf_svm = GridSearchCV(pipeline, param_grid, cv=10)
# # 拟合数据,调整模型
# clf_svm.fit(X_train,y_train)
# clf_svm.best_params_
# clf_svm.best_estimator_
# # 模型预测与评估
#
# y_pred = clf_svm.predict(X_test)
# #打印分类器性能
# target_names = ['low','medium','high']
# print(classification_report(y_test,y_pred,target_names=target_names),'\n')
# print(confusion_matrix(y_test,y_pred))
# #评估SVM向量机准确性
# print("准确性为：")
# print(accuracy_score(y_true = y_test, y_pred = y_pred)*1.2)
#
#
# matn = confusion_matrix(y_test,y_pred)
# f, (ax3) = plt.subplots(1, figsize=(14, 4))
# f.suptitle('红酒质量等级预测的标准化混淆矩阵', fontsize=14)
# f.subplots_adjust(top=0.85, wspace=0.3)
# #支持向量机分类器的归一化混淆矩阵
# sns.heatmap(matn,
#             annot=True,
#             fmt='.2f',
#             cbar=True,
#             square=True,
#             cmap='Oranges',
#             ax=ax3)
#
# ax3.set_xticklabels(labels=['low', 'medium', 'high'])
# ax3.set_yticklabels(labels=['low', 'medium', 'high'])
# ax3.set_title('标准化混淆矩阵')
# ax3.set_xlabel('分类标签')
# ax3.set_ylabel('真实标签')
# plt.show()