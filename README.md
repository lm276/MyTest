# MGStBot
A Multi-relational Graph-Based Twitter Account Stance and Bot Detection Benchmark.
### Introduction
[MGStBot]() is designed to achievement of tasks of Social media user stance detection and bot detection, MGStBot is the first standardized graph-based benchmark for stance detection and bot detection and built on the metadata of over 1.66 million users and 170 million tweets. For more details, please refer to the [MGStBot paper]().<br>

#### Statistics of stance detection datasets<br>
![Statistics of stance detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im1.png)<br>

#### Statistics of bot detection datasets<br>
![Statistics of bot detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im2.png)<br>

#### Distribution of Labels in annotations
![Statistics of bot detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im3.png)<br>

#### Relations in the MGStBot heterogeneous graph
![Statistics of bot detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im6.png)<br>

### Dataset Format
This dataset contains 'edge_index.pt','edge_type.pt','edge_weight.pt','features.pt','labels_bot.pt','labels_stance.pt'. See [here]() for a detailed description of these files.
### How to download MGStBot dataset

### Requirements
python 3.7<br>
torch：1.8.1+cu111<br>
torch_cluster-1.5.9<br>
torch_scatter-2.0.6<br>
torch_sparse-0.6.9<br>
torch_spline_conv-1.2.1<br>
torch-geometric 2.0.4<br>
pandas 1.3.4<br>
xgboost 1.6.2<br>

### How to run baselines

###  Performance of baseline methods on datasets
![Statistics of stance detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im4.png)<br>

###  Model accuracy of detection on MGStbot using different relations
![Statistics of stance detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im5.png)<br>

###  The performance of using different features on MGStBot
![Statistics of stance detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im7.png)<br>

###  Performance using different encoding models on MGStBot
![Statistics of stance detection datasets](https://github.com/lm276/MyTest/blob/main/pics/im8.png)<br>

### Citation
Please cite [MGStBot]() if you use the MGStBot dataset or this repository

### Literature used
[19] Shangbin Feng, Herun Wan, Ningnan Wang, and Minnan Luo. Botrgcn: Twitter bot detection with relational graph convolutional networks. Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, 2021. 1, 3, 5, 7, 11, 12<br>

[60] Kai-Cheng Yang, Onur Varol, Pik-Mai Hui, and Filippo Menczer. Scalable and generalizable social bot detection through data selection. In AAAI, 2020. 1, 3, 11, 12<br>

[34] Niyaz Jalal and Kayhan Zrar Ghafoor. Machine learning algorithms for detecting and analyzing social bots using a novel dataset. ARO-THE SCIENTIFIC JOURNAL OF KOYA
UNIVERSITY, 2022. 3, 11, 12<br>
