**通过opencv和numpy实现的数据增强模块。**

------

# RandAugment

- 论文地址：https://arxiv.org/pdf/1909.13719.pdf
- 开源代码github地址：https://github.com/heartInsert/randaugment
- **AutoAugment**的搜索方法比较暴力，直接在数据集上搜索针对该数据集的最优策略，其计算量很大。在**RandAugment**文章中作者发现，一方面，针对越大的模型，越大的数据集，使用**AutoAugment**方式搜索到的增广方式产生的收益也就越小；另一方面，这种搜索出的最优策略是针对该数据集的，其迁移能力较差，并不太适合迁移到其他数据集上。
- 在**RandAugment**中，作者提出了一种随机增广的方式，不再像**AutoAugment**中那样使用特定的概率确定是否使用某种子策略，而是所有的子策略都会以同样的概率被选择到，论文中的实验也表明这种数据增广方式即使在大模型的训练中也具有很好的效果。

## 测试使用

![](./images/原图.png)

```python
np.random.seed(2020)
image = cv2.imread(filename, flags=1)[:, :, [2, 1, 0]]
augment = RandAugment(size=image.shape[:-1], number=3)
image = augment(image)
```

![](./images/RandAugment1.png)

```python
np.random.seed(2020)
image = cv2.imread(filename, flags=1)[:, :, [2, 1, 0]]
augment = RandAugment(size=image.shape[:-1], number=3)
image = augment.shearY(image, 0.2)
```

![](./images/RandAugment2.png)

## 添加功能

- 原功能有14个，分别是“autocontrast“，“equalize“，“rotate“，“solarize“，“color“，“posterize“, “contrast“,  “brightness“，“sharpness“，“shearX“，“shearY“，“translateX“，“translateY“。

- 添加new_func：

  - self.transforms中加入"new_func"；

  - self.ranges中加入new_func的范围；

  - 定义new_func：

    ```python
    def new_func(self, img, m):
    	# 处理图像
    	return img
    ```

  - self.functions中加入new_func。

# MixUp

- 论文地址：https://arxiv.org/pdf/1710.09412.pdf

- 开源代码github地址：https://github.com/facebookresearch/mixup-cifar10

- **MixUp**是最先提出的图像混叠增广方案，其原理简单、方便实现，不仅在图像分类上，在目标检测上也取得了不错的效果。为了便于实现，通常只对一个batch内的数据进行混叠，在 **CutMix**中也是如此。

- 采用插值的思想：

  ![](./images/MixUp公式.png)

## 测试使用

```python
images, labels = [], []
for i in os.listdir(path1):
    images.append(cv2.resize(cv2.imread(path1 + i, flags=1)[:, :, [2, 1, 0]], (224, 224)))
    labels.append(np.array([1, 0]))
for i in os.listdir(path2):
    images.append(cv2.resize(cv2.imread(path2 + i, flags=1)[:, :, [2, 1, 0]], (224, 224)))
    labels.append(np.array([0, 1]))
images = np.asarray(images, dtype=np.float) / 255.
labels = np.asarray(labels, dtype=np.float)

augment = MixUp()
images, labels = augment(images, labels)
```

![](./images/MixUp1.png)

![](./images/MixUp2.png)