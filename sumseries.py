
#!/usr/bin/python
# -*- coding: UTF-8 -*-


def sumseries(n):
    data = 0
    a = 0 
    while (a <= n):
        data+=(a+1)* 2*10**(n-a)
        a+=1
    return data

a = 3
print (sumseries(a))