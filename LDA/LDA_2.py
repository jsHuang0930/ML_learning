import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
plt.rcParams['axes.unicode_minus']=False #plt中显示负号，避免乱码


# define converts(字典)
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

#LDA方法
def LDA_reduce_dimension(X, y, dim):
    # 输入：X为数据集(m*n)，y为label(m*1)，dim为目标维数
    # 输出：W 矩阵（n * dim）
    labels = list(set(y))
    # list():转化为列表；set():剔除矩阵y里的重复元素,化为集合的形式
    """
    eg:
        >>> a=[3,2,1,2]
        >>> set(a)
        {1, 2, 3} 
        >>> list(set(a))
        [1, 2, 3]

        >>> e=set(a)
        >>> type(e)
        <class 'set'> #集合
        >>> f=list(e)
        >>> type(f)
        <class 'list'>#列表
    """

    xClasses = {}  # 字典
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])  # list解析
    """
    x=[1,2,3,4]
    y=[5,6,7,8]
    我想让着两个list中的偶数分别相加，应该结果是2+6,4+6,2+8,4+8
    下面用一句话来写:
    >>>[a + b for a in x for b in y if a%2 == 0 and b%2 ==0]  
    """

    # 整体均值
    meanAll = np.mean(X, axis=0)  # 按列求均值，结果为1*n(行向量)
    meanClasses = {}

    # 求各类均值
    for label in labels:
        meanClasses[label] = np.mean(xClasses[label], axis=0)  # 1*n

    # 全局散度矩阵
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

    # 求类间散度矩阵
    # Sb=sum(len(Xj) * np.dot((uj-u).T,uj-u))  j=1...k
    # Sb=np.zeros((len(meanAll), len(meanAll) )) # n*n
    # for i in labels:
    #     Sb+= len(xClasses[i]) * np.dot( (meanClasses[i]-meanAll).T.reshape(len(meanAll),1),
    #                                     (meanClasses[i]-meanAll).reshape(1,len(meanAll))
    #                                )

    # 计算(Sw^-1)*Sb的特征值和特征矩阵
    eigenValues, eigenVectors = np.linalg.eig(
        np.dot(np.linalg.inv(Sw), Sb)
    )
    # 提取前dim个特征向量
    sortedIndices = np.argsort(eigenValues)  # 特征值排序
    W = eigenVectors[:, sortedIndices[:-dim - 1:-1]]  # 提取前dim个特征向量
    return W

    """
    np.argsort()
    eg:
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])
    Two-dimensional array:
    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])
    >>> np.argsort(x, axis=0)
    array([[0, 1],
           [1, 0]])
    >>> np.argsort(x, axis=1)
    array([[0, 1],
           [0, 1]])
    """


#if '__main__' == __name__:
def main():
    # 1.读取数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2.LDA特征提取
    W = LDA_reduce_dimension(X, y, 2)  # 得到投影矩阵
    newX = np.dot(X, W)  # (m*n) *(n*k)=m*k
    # 3.绘图
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(1)
    plt.scatter(newX[:, 0], newX[:, 1], c=y, marker='o')  # c=y,
    plt.title('自行实现的LDA')

    # 4.与sklearn自带库函数对比
    lda_Sklearn = LinearDiscriminantAnalysis(n_components=2)
    lda_Sklearn.fit(X, y)
    newX1 = lda_Sklearn.transform(X)
    plt.figure(2)
    plt.scatter(newX1[:, 0], newX1[:, 1], marker='o', c=y)
    plt.title('sklearn自带LDA库函数')
    plt.show()

main()
