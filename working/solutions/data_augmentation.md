# Data augmentation
There are two parts of data augmentation:
- waveforms: add uniform, gaussian, pink noise
- melspectrum: mixup

## Waveforms
### Uniform noise

### Gaussian noise

概率密度函数服从高斯分布的噪声。

### Pink noise

粉噪声（1/f 噪声）是噪声强度反比于频率的噪声。频率越高，粉噪声越小。

参考：https://www.zhihu.com/question/347692086

### White noise(没有加)

白噪声（white noise）在噪声曲线上是一条水平线。

## Melspectrum
### Mixup
reference: https://arxiv.org/pdf/1710.09412.pdf

**简介**

    大型深度神经网络是非常强大的，但其损耗巨大的内存以及对对抗样本的敏感性一直不太理想。mixup是一个简单地减缓两种问题的方案。本质上，mixup在成对样本及其标签的凸组合（convex combinations）上训练神经网络。这样做，mixup规范神经网络增强了训练样本之间的线性表达。

**优点**

    - 改进神经网络架构的泛化性能
    - 减少对错误标签的记忆，增加对样本的鲁棒性
    - 稳定生成对抗网络的训练过程

**实现**
参考：https://www.zhihu.com/question/308572298?sort=created

1. 对于输入的一个batch的待测图片images，我们将其和随机抽取的图片进行融合，融合比例为lam，得到混合张量inputs；
2. 第1步中图片融合的比例lam是[0,1]之间的随机实数，符合beta分布，相加时两张图对应的每个像素值直接相加，即 `inputs = lam*images + (1-lam)*images_random`；
3. 将1中得到的混合张量inputs传递给model得到输出张量outpus，随后计算损失函数时，我们针对两个图片的标签分别计算损失函数，然后按照比例lam进行损失函数的加权求和，即`loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)`；

```
i,(images,target) in enumerate(train_loader):
# 1.input output
images = images.cuda(non_blocking=True)
target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

# 2.mixup
alpha=config.alpha
lam = np.random.beta(alpha,alpha)
index = torch.randperm(images.size(0)).cuda()
inputs = lam*images + (1-lam)*images[index,:]
targets_a, targets_b = target, target[index]
outputs = model(inputs)
loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

# 3.backward
optimizer.zero_grad()   # reset gradient
loss.backward()
optimizer.step()
```




