[TOC]

## 1. 介绍

​	主要思路是用flask构建一个可以实时调用pytorch模型的地址，然后再将其部署到docker里，如果可以的话，再加上内网映射。



## 2.文件准备

首先准备文件目录如下：

- app           #一个自定义的目录
  - detectFlask.py                                #运行flask的主文件
  - Dockerfile                                     #用来build的dockerfile
  - mnist1d.py                                    #加载Torch模型的文件
  - mnist1d_detection_notzip.pkl           #预先训练好的模型



其中：

- detectFlask.py

```python
'''
此代码借鉴自参考链接1博主大大的，稍作修改
'''
PAGE = '''<!doctype html>
    <title>Chinese Text Detector</title>
    <h1>Chinese Text Detector</h1>
    <form action="" method=post enctype=multipart/form-data>
        <p>
         <label for="image">image</label>
         <input type=file name=file required>
         <input type=submit value=detect>
    </form>
    '''

from flask import Flask,request,jsonify
import cv2
import numpy as np
from mnist1d import detection,load_model    #这是后续导入自己模型的文件

app = Flask('Detector')

@app.route('/', methods=['GET', 'POST'])
# def index():
#     print("GO GO GO!!!")
#     return redirect(url_for('detector'))

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    print("GO GO!!")
    if request.method !='POST':
        return PAGE
    img = request.files['file']
    data = img.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)     
    net = load_model()
    result = int(detection(net=net,img=img))
    
    # print('inference time: ', time.time()-tic)

    return jsonify(msg='success', data={'result': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)  #这个5000有用的，debug关掉也可以
```

- DockerFile

```dockerfile
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

ENV LISTEN_PORT 5000
EXPOSE 5000

COPY ./mnist1d.py /app/
COPY ./mnist1d_detection_notzip.pkl /app/
COPY ./detectFlask.py /app/
WORKDIR /app

RUN pip install flask
RUN pip install opencv-python

CMD ["python", "detectFlask.py"]
```

```python
#这是上面那个dockerfile的注释
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel   #1.2这里有坑，注意避雷用自己版本的Torch

ENV LISTEN_PORT 5000                
EXPOSE 5000                                      #和detectflask文件选择的5000保持一致

COPY ./mnist1d.py /app/                          #注意mnist1d.py需要与文件同目录
COPY ./mnist1d_detection_notzip.pkl /app/        #权重
COPY ./detectFlask.py /app/						 #flask文件
WORKDIR /app                                     #选择app作为工作目录

RUN pip install flask                            #如果可以的话找个不用安装的镜像
RUN pip install opencv-python

CMD ["python", "detectFlask.py"]                 #运行
```

- mnist1d.py

```python
import torch
import torch.nn as nn

class d1Net(nn.Module):    #自己搭的1维卷积神经网络检测mnist数字，效果一般，主要是用来跑通用一下
    def __init__(self):
        super(d1Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2),  #in:784*1*b, out:392*10*b
            nn.MaxPool1d(kernel_size=3, stride=2),                               #in:392*10*b, out:196*10*b
            nn.ReLU(),
            nn.Conv1d(10, 20, 3, 2),                                              #in:196*10*b, out:98*20*b
            nn.MaxPool1d(kernel_size=3, stride=2),                                #in:98*20*b, out:49*20*b
            nn.ReLU(),
            nn.Conv1d(20, 40, 4)                                                   #input:49*20*b, 12*40*b
        )
        self.fc = nn.Sequential(
            nn.Linear(1800, 120),
            nn.Linear(120,84),
            nn.Linear(84,10)
        )
        
    def forward(self,x):
        x = x.view(-1,1,784)
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = x.view(-1,1800)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x

def load_model():     #加载模型函数
    net = d1Net()
    # net.load_state_dict(torch.load(r'D:\py\textDectectionDocker\mnist1d_detection_notzip.pkl'))      #绝对路径，用来docker前测试
    net.load_state_dict(torch.load(r'mnist1d_detection_notzip.pkl'))
    #torch.save(net.state_dict(), "mnist1d_detection_notzip.pkl",_use_new_zipfile_serialization=False) #保存不是zip的权重
    return net

def detection(net,img):          #测试函数
    x = net(torch.from_numpy(img).float())
    #print(x)
    return torch.max(x,1).indices[0]
```



## 3.制作镜像

​	准备好文件后，可以开始制作镜像，因为此文件中需要安装依赖，所以时间较长

```cmd
$cd xx/app
$docker build -t detector:v1.0 .
$docker run -p 3223:5000 -d --name detector detector:v1.0
```

> 这里的-d表示后台运行，在尝试时可以去掉以查看docker内响应情况

- 此时打开localhost:3223即可看到网页，提交图片检测即可得到输出结果，表示docker和flask的联动也成了



## 4.内网映射

​	这里选用的花生壳的内网映射（毕竟之前搞小程序的时候买了个域名、http映射、https映射还没用上呢）

- 应用类型选择-HTTPS
- 外网域名填上自己域名，端口填端口
- 内网主机ip:127.0.0.1，端口选择映射后的3223

映射完成后就可以通过访问域名到达网站了，这波flask-pytorch-docker-内网映射的闹剧也就此拉下帷幕。



## 5.踩坑指南

​	拉下帷幕……才怪，接下来介绍一下中间经历的部分弯弯绕绕吧，我内部标签可是有意识地被我迭代到了1.7，真的难受。

### 5.1 _use_new_zipfile_serialization=False

> 首先是我直接搬了博主大大的torch1.2版本出现的问题，因为我的环境是torch>1.6的，在1.6以后torch.save保存的模型是以zip格式保存的，1.2会报错。

- 解决办法：

  - 在1.6的环境保存一个不是zip格式的，重新制作镜像（下载一个1.6的我觉得会花更多时间，放弃）

  - ```python
    torch.save(net.state_dict(), "mnist1d_detection_notzip.pkl",_use_new_zipfile_serialization=False)
    ```

### 5.2 访问的网址问题

> flask文件内host=0.0.0.0，我直接浏览器一摸黑直接输入0.0.0.0:3223或者0.0.0.0:3223/detector之类的网址，反正就是访问不出来

- 解决办法：说实话我现在还是不知道0.0.0.0 v.s. 127.0.0.1的区别，虽然我每次输入都会有一个搜索词条出来，但是我……（算了，不说借口了，就是没看）
- 偶然在docker的客户端（我是在Windows）点了一个"OPEN IN BROWSE"，然后就打开了，我一开始还以为是端口传输问题，找了半天port的麻烦，建议可以直接从客户端点看看是哪个网址，虽然现在都是直接localhost:3223



### 5.3 / to /detector的跳转问题

> 博主的flask文件里面还是有从/ 跳转到/detector 的url的代码的，虽然当时为了找是哪里的引用，逐个搜索去了，但是还是在某次中出现问题了

- 解决办法：索性是不要这个路由了，直接注释掉了，详见代码



### 5.4 dockerfile的工作目录设置问题

> 在mnist1d.py文件里有导入权重这个语句，而本来是同目录的但是硬是导入不成功，说没有这个文件

- 解决办法：首先我去找了别人的例程，跑了个python读取同目录txt的测试，发现非常顺畅，原因竟然是因为？
  - 在dockerfile里设置WORKDIR，详见代码



## 6.总结

- 一看第五部分，怎么就这点坑，就算不算前面没有迭代标签，那起码也有6个坑才对，怎么就这？就这点东西不是三俩下就搞完了

- （还用问，还不是你菜，这个华华就是训啦）

- emmm，上面部分叙述由于本人真的很菜（不是甩锅句嗷）、同样最近也被各种事情逼着走，所以没时间了解之所以然，可能会给各位看官造成一些困扰，在此致歉！另外参考的博主写的比在下真的详细很多，可以直接看参考的链接，但是如果我的文章能给看官们一点点帮助，那就更好了。

- 总的来说，这波联动起码是达到我想要的结果了，记录一下，之后回来看也方便些，下一波联动：微信云托管。

  



## 7.参考链接

1.[如何将pytorch模型通过docker部署到服务器 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/159191983)

2.[Docker搞一个flask的image， 然后通过端口映射让别人可以访问 - rookiehbboy - 博客园 (cnblogs.com)](https://www.cnblogs.com/chengege/p/12675137.html)

