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