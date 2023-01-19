# pi_cnn
从0开始在树莓派上训练神经网络
## 一、概述

人工智能，神经网络一直是近些年计算机领域的热点。那么今天，我们就要在树莓派上部署一个开源的图像分类系统。我们会先训练一个卷积神经网络，然后你可以输入不同的图片，让它完成分类的工作。本期教程也会非常简单好用，即使你不懂神经网络这些只要按照教程去修改就可以做任何图像的分类，希望通过本期视频让大家切身感受到神经网络的魅力。这一期教程主要的内容可以分为两块，分别是树莓派官方系统上最新的tensorflow安装，以及神经网络的部署训练。对于新手来说可以按我们教程改相应的东西去跑图像分类，进阶玩家可以在后续利用这次教程安装的tensorflow和opencv去跑自己的项目。好了，开始吧！
```bash
下面要用到的资料（树莓派系统、数据集、网络代码、安装包）在微信公众号【树莓派爱好者基地】发送【图像分类】关键词就可以收到下载链接。
```

```bash
视频教程地址：
哔哩哔哩bilibili：树莓派爱好者基地

视频VLOG记录：
哔哩哔哩bilibili：玩派VLOG
```
## 二、教程内容
### 1、数据集介绍

这次用的数据集是神经网络做分类任务中非常常用的Fashion-MNIST数据集。该数据集有10种类别的共7万个不同商品的正面图片。其中训练集60000张，测试集10000张。图片格式为28x28的灰度图片，如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/68fd441a02294f92bcf8b03897d2d5fe.png#pic_center)
 数据集的分类如下表：
![在这里插入图片描述](https://img-blog.csdnimg.cn/9f6fd84adee84fb1a54c45beb373e864.png#pic_center)
### 2、系统介绍
树莓派系统这里使用树莓派官方的buster版本系统，注意请勿使用最新的官方系统（因为最新的官方系统bullseye版本内置的python版本为3.9，与tensorflow包不匹配），资料中已经给大家附带了树莓派系统。

### 3、树莓派换源

```bash
sudo nano /etc/apt/sources.list
```

原来的用#号注释掉，换成下面的

```bash
deb http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main contrib non-free rpi
deb-src http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ buster main contrib non-free rpi
```

```bash
sudo nano /etc/apt/sources.list.d/raspi.list
```

原来的用#号注释掉，换成下面的

```bash
deb http://mirrors.tuna.tsinghua.edu.cn/raspberrypi/ buster main ui
```

```bash
sudo apt-get update

sudo apt-get upgrade
```

### 4、安装tensorflow
TensorFlow是Google于2015 年发布的深度学习框架，是目前主流的深度学习框架，我们在这里利用TensorFlow实现卷积神经网络的实现。
下载资料中的tensorflow安装包（资料中有2.3版本和2.4版本，如果你的电脑有tensorflow，那么选择和电脑一样的版本，若没有，那随便选一个就行），然后**通过U盘拷贝到树莓派pi目录中**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/566fca6a1ef54a028347468c4816ba46.png#pic_center)
假如自己想下载其他版本的，按照如下规则和方法下载

```bash
https://github.com/lhelontra/tensorflow-on-arm/releases
```

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/e0efb65329bd417ba03a18d22dc8c78e.png#pic_center)

下载的时候要注意下载对应的版本：
（1）中间的CPXX，代表python版本，选错了无法安装。例如python3.5，应该选择CP35，pytho2.7就选择CP27
（2）树莓派2/3/4就选择结尾是armv7l.whl的。
（3）树莓派不适合训练模型，所以先在自己的电脑上训练模型，然后在树莓派里直接使用训练好的模型。要注意树莓派和自己电脑的tensorflow版本要一致，不然会出错

### 5、安装tensorflow

```bash
sudo pip3 install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```
安装的过程中可能会出现某些依赖安装失败，那么把他的下载地址复制出来，在自己电脑下载好，然后复制到PI目录下面，在用下面命令安装。（XXXXXXXXXXXXXXX替换成你下载的文件名称）

```bash
sudo pip3 install  XXXXXXXXXXXXXXX
```

如果用上面方法还是没办法下载成功，那么可以使用下面命令安装:

```bash
sudo apt-get install python3-包名
例如：sudo apt-get install python3-opencv
```

### 6、神经网络部署
注意：代码为开源代码，作者信息在代码中进行了保留，在此表示感谢。我们在该代码基础上做了删减和修改，让它更适用于树莓派。

代码介绍：该代码为开源代码，作者信息保留在了代码中。
data_split.py：数据集划分代码，如果就用本文数据集，那么不需要执行该代码。
train_cnn.py：构建以及训练网络代码。
test_model.py：测试网络识别准确率代码，可不执行。
window.py：图形化页面代码。

先下载一些库

```bash
sudo apt-get install python3-opencv
sudo apt-get install python3-pyqt5
```

打开THONNY，打开训练网络代码（train_cnn.py），运行即可。然后打开图形页面代码（window.py）就可以开始测试啦。
![在这里插入图片描述](https://img-blog.csdnimg.cn/bba5bf55bffb49bf82e35bd9b322147f.png#pic_center)

注意：如果想更换其他数据集，可以在网上下载好，然后用数据集划分代码（data_split.py）对把数据集分为训练集和测试集，然后按照代码中的注释修改训练网络代码（train_cnn.py）和图形页面代码（window.py）就可以啦。
