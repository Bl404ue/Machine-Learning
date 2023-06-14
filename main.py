from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from hmmlearn import hmm
import time
import numpy as np

# 加载手写数字数据集
digits = load_digits()
X = digits.data
y = digits.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化分类器
classifiers = [
    ('Perceptron', Perceptron()),
    ('k-Nearest Neighbors', KNeighborsClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Logistic Regression', LogisticRegression(multi_class='ovr', max_iter=5000)),
    ('Maximum Entropy', LogisticRegression(multi_class='multinomial', max_iter=5000)),
    ('Support Vector Machine', SVC()),
    ('Hidden Markov Model', hmm.GaussianHMM(n_components=10))
]

# 在训练集上训练并在测试集上进行预测，计算准确率和时间
results = {}
for clf_name, clf in classifiers:
    if clf_name == 'Hidden Markov Model':
        start_train_time = time.time()
        clf.fit(X_train)
        end_train_time = time.time()

        start_pred_time = time.time()
        y_pred = [np.argmax(clf.predict(seq.reshape(-1, 1))) for seq in X_test]
        end_pred_time = time.time()
    else:
        start_train_time = time.time()
        clf.fit(X_train, y_train)
        end_train_time = time.time()

        start_pred_time = time.time()
        y_pred = clf.predict(X_test)
        end_pred_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    train_time = end_train_time - start_train_time
    pred_time = end_pred_time - start_pred_time

    results[clf_name] = (accuracy, train_time, pred_time)

# 输出结果
for clf_name, (accuracy, train_time, pred_time) in results.items():
    print(f'{clf_name}:')
    print(f'  Accuracy: {accuracy}')
    print(f'  Training Time: {train_time} seconds')
    print(f'  Prediction Time: {pred_time} seconds')
    print()
