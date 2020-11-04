import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
plt.rcParams['axes.unicode_minus']=False #plt中显示负号，避免乱码

# iris共三种花卉种类{Iris-setosa,Iris-versicolor,Iris-virginica}

# LDA方法
def LDA_reduce_dimension(X, y, dim):
    # 输入：X为数据集(m*n)，y为label(m*1)，dim为目标维数
    # 输出：W 矩阵（n * dim）
    labels = list(set(y))
    # list():转化为列表；set():剔除矩阵y里的重复元素,化为集合的形式

    xClasses = {}  # 字典
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])  # list解析

    # 整体均值
    meanAll = np.mean(X, axis=0)  # 按列求均值，结果为1*n(行向量)
    meanClasses = {}

    # 求各类均值
    for label in labels:
        meanClasses[label] = np.mean(xClasses[label], axis=0)  # 1*n

    # 全局散度矩阵St
    St = np.zeros((len(meanAll), len(meanAll)))
    St = np.dot((X - meanAll).T, X - meanAll)

    # 求类内散度矩阵Sw
    # Sw=sum(np.dot((Xi-ui).T, Xi-ui))   i=1...m
    Sw = np.zeros((len(meanAll), len(meanAll)))  # n*n
    for i in labels:
        Sw += np.dot((xClasses[i] - meanClasses[i]).T, (xClasses[i] - meanClasses[i]))

    # 求类间散度矩阵Sb
    Sb = np.zeros((len(meanAll), len(meanAll)))  # n*n
    Sb = St - Sw

    # 计算(Sw^-1)*Sb的特征值和特征矩阵
    eigenValues, eigenVectors = np.linalg.eig(
        np.dot(np.linalg.inv(Sw), Sb)
    )
    # 提取前dim个特征向量
    sortedIndices = np.argsort(eigenValues)  # 特征值排序
    W = eigenVectors[:, sortedIndices[:-dim - 1:-1]]  # 提取前dim个特征向量
    return W

def main():
    # 读取数据集
    iris = load_iris()
    #X = iris.data[:,:2] #选取特征数据集的前两个数据特征
    X = iris.data #选取所有数据特征
    y = iris.target

    # LDA特征提取
    W = LDA_reduce_dimension(X, y, 2)  #得到投影矩阵
    print ("投影矩阵W：\n",W)
    newX = np.dot(X, W)  # (m*n) *(n*k)=m*k

    # 可视化图像
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
    plt.figure(1)
    plt.scatter(newX[:, 0], newX[:, 1], c=y, marker='o')
    plt.title('LDA分类可视化图')

    # 使用sklearn自带库函数与上面做比较
    lda_Sklearn = LinearDiscriminantAnalysis(n_components=2)
    lda_Sklearn.fit(X, y)
    newX1 = lda_Sklearn.transform(X)
    # 分成训练样本和测试样本，输出预测准确率
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    lda_Sklearn_test = LinearDiscriminantAnalysis()  #当不输入参数时，默认情况下是OVR方式
    lda_Sklearn_test.fit(X_train, y_train)
    print("预测准确率：",lda_Sklearn_test.score(X_test, y_test))
    plt.figure(2)
    plt.scatter(newX1[:, 0], newX1[:, 1], marker='o', c=y)
    plt.title('使用sklearn自带LDA库函数')
    plt.show()

main()
