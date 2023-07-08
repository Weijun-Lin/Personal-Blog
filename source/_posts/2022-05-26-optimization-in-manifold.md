---
title: 非线性优化的流形和局部参数化
mathjax: true
date: 2022-05-26
categories: 
- Math
tags:
- manifold
- 非线性优化
---

非线性优化普遍的解决思路是使用非线性最小二乘，非线性最小二乘可参考前一篇文章[“相机位姿估计”](https://weijun-lin.top/2022/03/13/2022-03-13-%20%E7%9B%B8%E6%9C%BA%E4%BD%8D%E5%A7%BF%E4%BC%B0%E8%AE%A1%E4%B8%8E%E7%82%B9%E4%BA%91%E4%BC%98%E5%8C%96/#%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98)。在很多优化过程中，如果优化变量本身具有一定的约束，这样的优化将变得十分复杂，所以如果能将约束问题转换为无约束的这可以带来极大的便利。在高翔博士的《SLAM 十四讲》中，其中的 BA 优化便利用李群李代数构造旋转矩阵的无约束优化问题，这在博客[“相机位姿估计”](https://weijun-lin.top/2022/03/13/2022-03-13-%20%E7%9B%B8%E6%9C%BA%E4%BD%8D%E5%A7%BF%E4%BC%B0%E8%AE%A1%E4%B8%8E%E7%82%B9%E4%BA%91%E4%BC%98%E5%8C%96/#%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98)中也比较详细介绍了。这里记录一下对流形优化新的思考。

<!-- more -->

非线性最小二乘一般使用库 [Ceres Solver](ceres-solver.org) 构建，它提供了将变量转换到其流形上的方法，称之为局部参数化（[Local Parameterization]([Modeling Non-linear Least Squares — Ceres Solver (ceres-solver.org)](http://ceres-solver.org/nnls_modeling.html#_CPPv4N5ceres21LocalParameterizationE))），这个参数化过程便是其所在切空间到其流形的映射表示。

## 流形的不严谨介绍

流形是一个空间，它表示了变量的本质，其所在的真实空间。比如三维空间的一条曲线，这条曲线所在的空间是三维的，即它嵌入在三维空间中，但实际上它是一条曲线，曲线只有一个自由度，即我们可以将曲线参数化表达为：

$$
l = 
\left\{
\begin{align}
x(t) \\
y(t) \\
z(t)
\end{align}
\right.
$$

很清楚的看到表达一条曲线其实只需要一个变量，即它本质上是一维的。这个一维空间便是它的流形空间，它流形是一维的。类似的，三维空间中的曲面，是一个二维流形。流形只能嵌入到比它维数更大的空间中。流形的维度和其变量所在的切空间有关，切空间是指变量某个取值处的切向量。拿曲线来说，三维空间曲线某点的切空间就是其切线，对于曲面则是其切平面。切空间满足欧式空间的性质，在切空间的改变量需要投影回流形，完成迭代更新。

## 流形上的优化

之所以要将变量参数化到其流形上，是为了构建无约束的优化。在非线性优化中，比如经典的高斯牛顿法，本质上都是构建迭代：
$$
x_{k+1} = x_k + \Delta x_k
$$

而这个迭代在变量自身有约束的情况下不一定成立，即更新后 $x_{k+1}$ 不满足约束条件，比如约束 x 在某一条曲线上，当它执行简单的加法更新后它并不一定会在曲线上，但转移到流形上则不一样了。**流形上的优化（局部参数化）可以理解为将 $\Delta x_k$ 转变为 $x_k$ 切空间上的改变量，更新变量后投影回原空间得到 $x_{k+1}$**。Ceres 中的局部参数化也就是这个过程，其本质是重载了优化迭代中的“加法”，可以表示为：
$$
\boxplus (x,\Delta)
$$
在欧式空间中，$\boxplus (x,\Delta) = x + \Delta$，在李代数中 $\boxplus (x,\Delta) = \exp(\Delta)x$，$\Delta$ 是其切空间上的改变量，$\exp(\Delta) x$ 将其，然后更投影回流形并更新 $x$。这里李群通过矩阵乘法更新。

通过与高斯牛顿法比较，改变量变成了切空间上的量，所以需要对这个改变量（扰动）求导，这也就李导数中的扰动求导，即对扰动求导。

## 方向向量求导

这里给一个具体例子，方向向量的局部参数化和对扰动求导，方向向量指的是表示一个指向的向量，为单位向量。三维空间内的方向向量只有两个自由度，它构成的集合是球心在原点的单位球。方向向量 $n$ 的扰动求导可以表示为：
$$
n + \epsilon_{3\times 1} = \exp_{so3}(T_{3\times 2}\delta_{2\times 1})n
$$
扰动为 $\delta$ 是一个二维向量，是在其切空间上的改变量，$T_{3\times 2}$ 为 $n$ 的切空间，即切平面，通过 $T_{3\times 2}\delta_{2\times 1}$ 重新映射到三维欧式空间，为切平面上改变量在三维空间的表示，最后通过 $\exp_{so3} \times n$ 投影到流形（单位球）上，这里将 $T\delta$ 理解为旋转向量。$n$ 的改变量对扰动的求导可以表示为：
$$
\begin{align}
\frac{\partial n}{\partial \delta} 
&= \frac{\partial \epsilon}{\partial \delta} \\
&= \frac{ \exp_{so3}(T_{3\times 2}\delta)n - n}{\delta} \\
&= \frac{I + (T_{3\times 2}\delta)^{\wedge}n - n}{\delta} \\
&= \lim_{\delta\rightarrow 0}\frac{(T\delta)^{\wedge}n}{\delta} \\
&= \frac{\partial (T\delta)^{\wedge}n}{\partial(T\delta)}
\frac{\partial T\delta}{\partial \delta} \\
& = -n^{\wedge}T
\end{align}
$$

最后迭代更新为：
$$
n' = \exp_{so3}(T_{3\times 2}\delta_{2\times 1})n
$$
