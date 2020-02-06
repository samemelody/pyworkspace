
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

max=7
data = 1
arr = np.zeros((max,max),dtype=int)
circlecnt = (max)//2
#圈数
a = 0
while a < circlecnt:
    i = a
    j = a
    while i< max - a:
        arr[i][j] = data
        data+=1
        i+=1
    i-=1
    j+=1
    arr[i][j] = data
    while j < max -a:
        arr[i][j] = data
        data+=1
        j+=1
    j-=1
    i-=1
    arr[i][j] = data
    while i > a:
        arr[i][j] = data
        data+=1
        i-=1
    while j > a:
        arr[i][j] = data
        data+=1
        j-=1
    a+=1
#当螺旋阵为奇数时需要额外填中心点
if max %2 ==1:
    arr[a][a] = data    
    
print (arr)