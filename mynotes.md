## SciPy
SciPy 是Python 中用于科学计算的函数集合。它具有线性代数高级程序、数学函数优化、信号处理、特殊数学函数和统计分布等多项功能。scikit-learn 利用SciPy 中的函数集
合来实现算法。对我们来说，SciPy 中最重要的是scipy.sparse：它可以给出稀疏矩阵（sparse matrice），稀疏矩阵是scikit-learn 中数据的另一种表示方法。如果想保存一个大
部分元素都是0 的二维数组，就可以使用稀疏矩阵：
```python
sparse_matrix = sparse.csr_matrix(eye);
```
