# GNN-EA

## 总体架构

![image](https://github.com/TimDinggw/GNN-EA/blob/main/fig/framework.png)

## 运行环境

- python 3.9.12
- torch 1.12.0
- torch-cluster 1.6.0
- torch-geometric 2.4.0
- torch-scatter 2.1.0
- torch-sparse 0.6.16
- torch-spline-conv 1.2.1

## 数据集

解压 https://github.com/TimDinggw/GNNinEAFramework 中 data.zip 文件
- DBP15K from HGCN
- SRPRS from EvalFramework. 另外，我们利用BERT+Linear(768,300)为其建立了与HGCN中类似的300维的实体名称初始化词典。


## 运行方法

主要参数：

```
python run.py
--encoder default="GCN-Align"
--hiddens default="300,300,300" (including in_dim and out_dim)
--ent_init default='random', choices=['random', 'name']
--skip_conn default='none', choices=['none', 'highway', 'concatall', 'concat0andl', 'residual', 'concatallhighway']
--activation default='none', choices=['none', 'elu', 'relu', 'tanh', 'sigmoid']
```

## 参考代码

EAkit(https://github.com/THU-KEG/EAkit)