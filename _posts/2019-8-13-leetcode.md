---
title: leetcode
data: 2019-8-13
categories: leetcode
---

##### 48. 旋转图像

给定一个 *n* × *n* 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

```python
# 将矩阵旋转90度相当于对矩阵先转置，然后对每行反转
# swap 用于交换两个成员

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        length = len(matrix)
        for i in range(length):
            for j in range(i+1, length):
                tmp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = tmp
        for i in range(length):
            matrix[i] = matrix[i][::-1]
```

