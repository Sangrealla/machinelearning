## NumPy
NumPy 是Python 科学计算的基础包之一。它的功能包括多维数组、高级数学函数（比如线性代数运算和傅里叶变换），以及伪随机数生成器。  
在scikit-learn 中，NumPy 数组是基本数据结构。scikit-learn 接受NumPy 数组格式的数据。你用到的所有数据都必须转换成NumPy 数组。NumPy 的核心功能是 ndarray 类，即多维（n 维）数组。数组的所有元素必须是同一类型。NumPy 数组如下所示：  
```python
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
```

## SciPy
SciPy 是Python 中用于科学计算的函数集合。它具有线性代数高级程序、数学函数优化、信号处理、特殊数学函数和统计分布等多项功能。  
scikit-learn 利用SciPy 中的函数集合来实现算法。对我们来说，SciPy 中最重要的是scipy.sparse：它可以给出稀疏矩阵（sparse matrice），稀疏矩阵是scikit-learn 中数据的另一种表示方法。如果想保存一个大部分元素都是0 的二维数组，就可以使用稀疏矩阵：
```python
from scipy import sparse
# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("NumPy array:\n", eye)
eye
# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
```
通常来说，创建稀疏数据的稠密表示（dense representation）是不可能的（因为太浪费内存），所以我们需要直接创建其稀疏表示（sparse representation）。  
下面给出的是创建同一稀疏矩阵的方法，用的是COO 格式：  
```python
from scipy import sparse
import numpy as np
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))
```
## matplotlb
matplotlib 是Python 主要的科学绘图库，其功能为生成可发布的可视化内容，如折线图、直方图、散点图等。将数据及各种分析可视化，可以让你产生深刻的理解，而
我们将用matplotlib 完成所有的可视化内容。  
在Jupyter Notebook 中， 你可以使用%matplotlib notebook 和%matplotlib inline 命令，将图像直接显示在浏览器中。我们推荐使用%matplotlib notebook 命令，它可以提供交互环境（虽然在写作本书时我们用的是%matplotlib inline）
```python
%matplotlib inline
import matplotlib.pyplot as plt
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
```
## pandas
pandas 是用于处理和分析数据的Python 库。它基于一种叫作DataFrame 的数据结构，这种数据结构模仿了R 语言中的DataFrame。简单来说，一个pandas DataFrame 是一张表格，类似于Excel 表格。  
pandas 中包含大量用于修改表格和操作表格的方法，尤其是可以像SQL 一样对表格进行查询和连接。NumPy 要求数组中的所有元素类型必须完全相同，而pandas 不是这样，每一列数据的类型可以互不相同（比如整型、日期、浮点数和字符串）。pandas 的另一个强大之处在于，它可以从许多文件格式和数据库中提取数据，如SQL、Excel 文件和逗号分隔值（CSV）文件。
```python
import pandas as pd
from IPython.display import display
# 创建关于人的简单数据集
data = {'Name': ["John", "Anna", "Peter", "Linda"],  
        'Location' : ["New York", "Paris", "Berlin", "London"],  
        'Age' : [24, 13, 53, 33]  
        }
data_pandas = pd.DataFrame(data)
# IPython.display可以在Jupyter Notebook中打印出“美观的”DataFrame
display(data_pandas)
```


















