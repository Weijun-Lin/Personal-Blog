---
title: 飞行模拟机的自动视景融合校正系统，省级大学生创新创业项目，效果展示
date: 2020-04-01
tags:
- 几何校正
categories:
- Computer Science
- CV&CG&DIP
- DIP
---

> 项目环境：Python 3.7 
>
> 开发环境：Visual Studio Code
>
> 库依赖：
>
> - matplotlib
> - numpy
> - opencv (cv2)

<!-- more -->

## Cubic Spline Interpolation 三次样条插值

> 输入N个控制点，通过样条插值模拟出通过这N个点的光滑曲线

### 原理参考

主要原理可参考：*https://www.cnblogs.com/liuyunfeifei/articles/3110140.html*

### 作用

可作为几何校正曲线模拟函数（暂时几何校正还使用贝塞尔曲线模拟）

还可作为边缘融合重合地带的亮度调整函数

### 代码简介

#### 三次曲线类

```python
class CubicLine:
    '''
    三次曲线类 
    形如：y = a + b(x-xi) + c(x-xi)^2 + d(x-xi)^3
    '''
    def __init__(self, a, b, c, d, xi):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.xi = xi

    def getResult(self, x):
        return self.a + self.b*(x - self.xi) + self.c*((x - self.xi)**2) + self.d*((x - self.xi)**3)
```

#### 三次样条

```python
class CubicSpline:
    '''
    三次样条类，通过给定的控制点形成曲线
    @Attrs:
        points: 控制点数组
        H: 步长数组
        m: 求解数组
        cubicLines: 最后的三次样条分段函数
    '''
    def __init__(self, points: np.ndarray):
        ''' 接受一个节点数组 '''
		pass
    
    def setPoints(self, points: np.ndarray):
        ''' 重新更新控制点(节点数组) '''
		pass      

    def solve(self):
        ''' 求解  M * m = b'''
		pass
    
    def getS(self):
        ''' 获取曲线段分段表达式 '''
		pass
```

### 演示效果

> 通过matplotlib展示构建的样条函数 并通过设置相应的回调函数实现交互输入

<img src="/assets/ArticleImg/2020/cubicSplineInterpolation.png" style="zoom:50%;" />

## Edge Fusion 边缘融合

> 想要两个投影机的投影图像没有边缘界线，必然需要重叠中间一部分，这是**生成无缝拼接的需要**
>
> 需要构建合适的边缘融合函数以此抵消重合带的亮度改变

### 原理介绍

<img src="/assets/ArticleImg/2020/edgeFusion1.png" style="zoom:50%;" />

中间的融合带不加处理的化必定会造成亮度叠加，会造成亮度突变

如图：在融合带边缘尤为明显

- 原图：从左到右亮度逐渐增加的灰度图

    <img src="/assets/ArticleImg/2020/edgeFusion3.png" style="zoom:50%;" />

- 直接叠加：中间亮度突变

    <img src="/assets/ArticleImg/2020/edgeFusion4.png" style="zoom:50%;" />

所以我们需要将左通道和右通道的融合区域进行预处理之后然后叠加

刚刚开始我的思路是左边融合带和右边融合带亮度都衰减为原来的50%那么叠加起来就和原图一样了，这里有个错误的判断，这个前提是两个投影机的亮度输出必须完全一致，但在实际里，两个投影机的输出亮度都是不同的，所以融合函数是简简单单的常数函数是不行的，必须是要满足渐变这个前提条件

#### 不同的投影机亮度输出的效果图

> 这里采用左通道亮度为原图的0.7倍，而右通道为原图的1.3倍

可以看出效果特别不好甚至是没有任何改善

<img src="/assets/ArticleImg/2020/edgeFusion5.png" style="zoom:50%;" />

当使用比较常见的融合分段函数时，效果如下，这样就不存在亮度跳变了

<img src="/assets/ArticleImg/2020/edgeFusion6.png" style="zoom:50%;" />

融合函数具体为：

<img src="/assets/ArticleImg/2020/edgeFusion7.png" style="zoom:100%;" />

改变参数的效果：可以看出**a控制的是分段函数连接点，p控制的幂函数的次数**

右侧函数值就是**1-左侧函数值**

<img src="/assets/ArticleImg/2020/edgeFusion8.png" style="zoom:100%;" />

### 代码概览

```python
def edgeFusion(img: np.ndarray, brightL, brightR, f):
    '''
    @Args
        img: 输入图像 此示例为将输入图像分割为左2/3和右2/3中间融合带1/3来演示不同融合函数效果
        brightL: 左侧亮度调整参数
        brightR: 右侧亮度调整参数
        f：融合函数
    @Return
        newimg: 融合完成的图像
    '''
    rows, cols = img.shape[0:2]
    bk = np.zeros(img.shape)
    img = img.astype(bk.dtype)
    tape = 1/3  # 融合带占图片的1/3

    xL = int(cols * (1/2 + tape/2)) # 左投影的边界
    xR = int(cols * (1/2 - tape/2)) # 右投影的边界
    tapeL, tapeR = xR, xL
    
    imgL, imgR = img[:, :xL].copy(), img[:, xR:].copy()

    # 左侧图像处理
    for i in range(tapeL, xL):
        value = f((i - tapeL)/(tapeR - tapeL))
        value = 1 - value
        imgL[:, i] = imgL[:, i]*value

    # 右侧图像处理
    for i in range(0, tapeR - tapeL):
        value = f(i/(tapeR - tapeL))
        imgR[:, i] = imgR[:, i]*value

    bk = bk.astype(np.uint8)
    imgL = imgL.astype(np.uint8)
    imgR = imgR.astype(np.uint8)
    # 叠加图像
    bk[:, :xL] = cv2.addWeighted(bk[:, :xL], 1, imgL, brightL, 0)
    bk[:, xR:] = cv2.addWeighted(bk[:, xR:], 1, imgR, brightR, 0)

    return bk.astype(np.uint8)
```

### 彩色图像演示

- 原图以及常数融合函数 左通道亮度0.7;右通道亮度1.3

    <img src="/assets/ArticleImg/2020/edgeFusion9.png" style="zoom:100%;" />

- 分段融合函数：a = 0.7 p = 2

    <img src="/assets/ArticleImg/2020/edgeFusion10.png" style="zoom:50%;" />

演示中窗口设置了滑条可以实时调整融合函数参数以便交互

## Geometric correction 几何校正

> 从原图到投影机到幕布会发生变换，所以我们需要对原图进行反变换来使最终图像显示正常

### 原理介绍

图像的连续的，我们只要确定四条边界就可以确定最终的图像，这就需要我们可以对边界进行任意改变，这就需要通过控制点改变边界以生成曲线

生成四条边界曲线后，然后将原图像映射到新的图像区域中即可

#### 边界曲线

这里采用的是三阶贝塞尔曲线，通过四个控制点就可以生成任意的三阶函数曲线如下图（是一个动图PDF查看不了）

![](/assets/ArticleImg/2020/transform1.gif)

#### 划分网格

需要将原图划分为N个矩形区域，如图：

<img src="/assets/ArticleImg/2020/transform2.png" style="zoom:80%;" />

#### 网格点重映射

然后通过调整控制点生成新的边界点，然后通过以下公式更新网格点

<img src="/assets/ArticleImg/2020/transform3.png" style="zoom:100%;" />

新的网格点如下：**蓝色点为控制点**

<img src="/assets/ArticleImg/2020/transform4.png" style="zoom:80%;" />

#### 纹理映射

然后只需要对每一个区域做一个纹理映射即可，这里用的是opencv带的仿射变换，但会出现最后拼接是出现锯齿或者是亮条的缺点，后期可以直接使用openGL或者direct3D直接进行纹理变换

### 代码概览

> - bezier.py：贝塞尔曲线
> - copyToQuadrangle.py：纹理映射函数
> - transform.py：旧的几何变换demo
> - test.py：新的几何变换demo

#### 三阶贝塞尔

> 类定义在文件bezier.py中

```python
import numpy as np

class Bezier:
    ''' 提供四个控制点和一个分段数 返回三阶贝塞尔曲线点集 '''

    def __init__(self, P0, P1, P2, P3):
        self.P0 = np.array(P0).reshape(1,2)
        self.P1 = np.array(P1).reshape(1,2)
        self.P2 = np.array(P2).reshape(1,2)
        self.P3 = np.array(P3).reshape(1,2)

    def getPoints(self, numbers):
        res = []
        for i in np.linspace(0, 1, numbers):
            res.append(self.getDetailPos(i))

        return np.array(res).reshape(-1, 2).astype(np.int32)

    def getDetailPos(self, t):
        return ((1 - t)**3)*self.P0 + 3*t*(1 - t)*(1 - t)*self.P1 + 3*t*t*(1 - t)*self.P2 + (t**3)*self.P3
```

#### 纹理映射

> 有两个纹理映射
>
> 1. 原图是任意四边形映射到目标图的任意四边形:copyToQuadrangle
> 2. 原图是规则矩形映射到目标图的任意位置:copyToQuadrangleWithRegular

```python
def copyToQuadrangle(srcImg: np.ndarray, tarRect1, desImg: np.ndarray, tarRect2):
    '''
    将 srcImg 的指定四边形区域 复制到desImg的指定区域
    出现边缘锯齿
    但内部整合效果好
    @Param:
        desImg: 目标图像
        srcImg: 源图像
        tarRect1: 目标四边形区域，左上角顺时针
        tarRect2: 源四边形区域
    @Return:
        新的图像        
    '''
    srcRows, srcCols = srcImg.shape[0:2]
    desRows, desCols = desImg.shape[0:2]
    tarRect1 = np.float32(tarRect1)
    tarRect2 = np.float32(tarRect2)

    # 首先创建目标掩码
    # 将整个图像变换到目标区域来创建掩码
	pass
    # 获取变换后的 ROI 图像
	pass
    # 处理目标图像 将需要重新贴图的地方置 0
    pass
    # 合并
    return cv2.add(midDesImg, tarImg)

def copyToQuadrangleWithRegular(srcImg: np.ndarray, tarRect1: np.ndarray, desImg: np.ndarray, tarRect2):
    '''
    将 srcImg 的指定四边形区域 复制到desImg的指定区域
    对目前项目 只需要从规则矩形（可以直接提取ROI）仿射至任意四边形即可
    可采用此函数避免锯齿
    但存在边缘高亮
    @Param:
        desImg: 目标图像
        srcImg: 源图像
        tarRect1: 目标四边形区域，左上角顺时针
        tarRect2: 源四边形区域
    @Return:
        新的图像       
    '''
    srcRows, srcCols = srcImg.shape[0:2]
    desRows, desCols = desImg.shape[0:2]
    # 获取原矩形宽度和高度
	pass
    # 获取ROI图像
	pass
    # 仿射变换
	pass
```

### 效果展示

- 变换前

    <img src="/assets/ArticleImg/2020/transform5.jpg" style="zoom:100%;" />

- 变换后 `copyToQuadrangle` 纹理映射 这里边缘出现了锯齿

    <img src="/assets/ArticleImg/2020/transform7.png" style="zoom:100%;" />

- 变换后 `copyToQuadrangleWithRegular`纹理映射 可以看到有亮条

    <img src="/assets/ArticleImg/2020/transform6.png" style="zoom:100%;" /> 

