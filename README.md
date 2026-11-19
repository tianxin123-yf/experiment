实验四：MLP 多层感知机
田鑫202410315035人工智能241
一、实验目的
1.理解多层感知机（MLP）的基本结构，掌握全连接层、激活函数、多分类交叉熵损失、随机梯度下降优化的原理。
2.学会使用 PyTorch 搭建简单神经网络，完成数据集加载、前向传播、反向传播、参数更新以及模型评估的完整流程。
3.利用 MLP 在 MNIST 手写数字数据集上完成图像分类，观察模型损失与分类准确率的变化规律。
4.熟悉 JupyterLab 的使用，掌握模型训练、测试、评估的代码编写与调试方法。
二、实验过程
搭建实验环境，安装 PyTorch、torchvision 等依赖包；

<img width="1107" height="322" alt="b0fbb9a8c509e120f0d8a78b0be1c59" src="https://github.com/user-attachments/assets/cdee51e0-956f-449f-8eb4-c69821e4f6c5" />

<img width="1108" height="299" alt="image" src="https://github.com/user-attachments/assets/92d02ca3-d06d-4b91-8e93-5840c2a0968b" />

加载 MNIST 数据集，构造训练集与测试集的数据加载器。

<img width="1149" height="496" alt="image" src="https://github.com/user-attachments/assets/97b5c0ac-3731-4fec-b31d-57f7b043f244" />

<img width="962" height="156" alt="image" src="https://github.com/user-attachments/assets/09e5d1e4-ebf3-436c-b215-ab7a887c0dee" />

定义 MLP 网络，包含输入展平、一层隐藏全连接层搭配 ReLU 激活函数，输出层得到十个类别的得分。
<img width="1108" height="442" alt="image" src="https://github.com/user-attachments/assets/c5296661-8b1c-422c-9046-d6f0b9b2cb08" />

<img width="1032" height="176" alt="image" src="https://github.com/user-attachments/assets/93b6d1a4-8e72-403f-8dc9-05c14d74b847" />

设置超参数：批次大小、训练轮数、学习率，选用交叉熵作为损失函数、SGD 作为优化器。

<img width="1107" height="282" alt="image" src="https://github.com/user-attachments/assets/8c59e088-9ea1-4b0e-b127-a5a73e337579" />

训练阶段将模型切换为训练模式，遍历训练集，每一批数据完成前向传播、损失计算、反向传播与参数更新；每一轮训练结束后，将模型切换为评估模式，在关闭梯度计算的条件下对测试集进行预测，统计测试损失和分类准确率并输出。

<img width="1107" height="428" alt="image" src="https://github.com/user-attachments/assets/f3c17c04-fbed-440a-9786-3285a7df5bb0" />

训练全部结束后，观察准确率是否逐步上升、损失逐步下降，分析模型的收敛效果。

<img width="898" height="516" alt="image" src="https://github.com/user-attachments/assets/6f497016-6e99-4941-ab1d-9ce4332bb9f6" />

观察发现：随着训练轮次增加，测试损失持续下降，准确率稳步上升至 0.8924，模型正常收敛。训练前期指标变化较快，后期提升放缓，达到当前结构的性能上限，无明显过拟合。该浅层 MLP 表达能力有限，后续可通过调整网络结构、优化器与超参数进一步提高精度。

三、遇到的问题及解决方式
①问题：运行代码提示 ModuleNotFoundError: No module named 'torch'
原因：当前 Jupyter 内核没有安装 PyTorch。
解决：使用 %pip 在 notebook 内安装，安装完成后重启内核，再运行导入代码。
②问题：单元格标记 [*]，长时间没有输出
原因：CPU 训练速度慢，批次设置过大；或是代码缩进错误，测试部分没有放在 epoch 循环内；也可能是正在下载数据集。
解决：调小 batch_size；检查缩进，保证测试评估代码在 epoch 循环内部；耐心等待数据集下载。
③问题：模型准确率偏低，收敛缓慢
原因：学习率不合适、网络结构过于简单。
解决：适当调整学习率；也可以增加隐藏层神经元数量或者增加隐藏层，提升模型拟合能力。

四、实验心得
本次实验让我完整走完神经网络分类的全部流程，不再只停留在理论层面。我认识到搭建网络不只是写代码，环境、缩进、参数、内核匹配任何一处出错，程序都无法正常运行。
训练过程中我发现，超参数对结果影响很大：批次大小、学习率都会改变模型收敛的速度和最终精度。同时，model.eval() 和 torch.no_grad() 的作用十分关键，可以减少不必要的梯度计算，提升测试效率。
这次实验也让我意识到深度学习调试比较繁琐，遇到报错不能急于跳过，要从报错信息定位根源。后续我会继续尝试修改网络结构、更换优化器，进一步体会不同设置对分类效果带来的影响。
