# coding=gbk

from bs4 import BeautifulSoup
import requests
from urllib.request import urlretrieve
import os
import time


turl = "http://xxx.net/"

theader = {
                "User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

req = requests.get(url = turl,headers = theader)
req.encoding = 'utf-8'
html = req.text
bf = BeautifulSoup(html, 'html.parser')
ls1 = []
#targets_url = bf.find_all(class_='item-img')
#ls1 = list(set( bf.find_all("a")))
for item in bf.find_all("a"):
    tempurl = item.get('href')
    if tempurl != 'index.html' and 'html' in tempurl:
        ls1.append(tempurl)
  #  print(item)
print('*****************')    
indexlist = list(set(ls1))
print(indexlist)

def getsuburl(list,urlpath):
    for item in list:
        time.sleep(1)
        req = requests.get(urlpath+item,headers = theader)
        req.encoding = 'utf-8'
        html = req.text
        df = BeautifulSoup(html, 'html.parser')
        lsxx = []
        for ggg in  df.find_all('a'):
            eee = ggg.get('href')
            if eee != 'index.html' and 'html' in eee and eee not in indexlist:
                lsxx.append(eee)
        if len(lsxx) > 0:
            getsuburl(lsxx, urlpath)
        else:
            list+lsxx;
        print('indexlist',indexlist)


getsuburl(indexlist,turl)
print('uio',indexlist)


