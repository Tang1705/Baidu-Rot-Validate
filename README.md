# Baidu-Rot-Validate


>目标：

识别图片角度，推算出对应滑动距离，模拟滑动。

>实现：

 1. 获得原始图片数据集：循环访问百度搜索页面从而进入百度安全验证页面，抓取图片1500张，获得大量重复图片。对这些图片进行筛选，获得不重复图片144张；

![在这里插入图片描述](https://img-blog.csdnimg.cn/5124b0ad94494ba8b54c8dca0b7e6c16.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAVGFuZzU2MTg=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

2. 制作数据集：生成各个角度图片模型，每个不同的图片均有360张不同角度的照片，标记正向图，根据现有图片名称序列计算不同角度照片相对于正向图片的相对角度，重新命名从而建立数据集；

![在这里插入图片描述](https://img-blog.csdnimg.cn/17cf39e2bd114730b196357bff83c45d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAVGFuZzU2MTg=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)![在这里插入图片描述](https://img-blog.csdnimg.cn/1eba81026b4749368832d211bf0f2d37.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAVGFuZzU2MTg=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

3. 生成规则：根据滑动距离与图片旋转角度的关系形成滑动规则（按照滑动距离 $o=angle \times 360 \times b$ 的规则来预测角度是不准确的，所以通过在安全验证页面逐单位距离滑动记录角度与滑动距离的关系）；
4. 模型训练：利用神经网络（考虑模型、数据集的大小以及模型的感受野）在数据集上进行训练，直接预测需要滑动的距离。由于百度安全验证的角度并不是整数变化的，滑动距离与角度的变化也不是一一对应的，因此相对于预测角度而言，直接预测滑动距离更加准确、便捷。且滑动距离的类别相比较于而言要少，从而使得模型的参数也更少。此外，因为模型的目的是能足够准确地预测滑动距离，从而使得自动化程序模拟验证，因此，应该使得模型尽可能多地在现有数据集上学习。综上，考虑到真实场景的验证图片与获得的图片存在一定的差异（即使是相同的图片，也会受到不同程度的噪声干扰，如水印等），不再对数据集进行训练集与测试集的划分。

>性能：

推理快、预测准（基本在三次验证中可通过，根据实际操作和 b 站视频来看，人为通过安全验证的难度也相对较大）

https://www.douyin.com/video/7070526136115105054?modeFrom=userPost&secUid=MS4wLjABAAAA1Yrnw5gwCNI_5nHTEeJtXCcSkkqajKIfspcXvw6Oxkg

> 代码

欢迎大家 star 和提 issue
