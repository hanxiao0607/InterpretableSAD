[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/InterpretableSAD/blob/main/LICENSE)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhanxiao0607%2FInterpretableSAD&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# InterpretableSAD: Interpretable Anomaly Detection in Sequential Log Data
A Pytorch implementation of [InterpretableSAD](https://ieeexplore.ieee.org/document/9671642).

## Abstract
Anomaly detection in sequential data is a common data analysis task as it contributes to detecting critical information, such as malfunctions of systems. However, due to the scarcity of the anomalies, the traditional supervised learning approaches cannot be applied for anomaly detection tasks. Meanwhile, most of the existing studies only focus on identifying the anomalous sequences and cannot further detect the anomalous events in a sequence. In this work, we present InterpretableSAD, an interpretable sequential anomaly detection framework that can achieve both anomalous sequence and fine-grained event detection. Given a set of normal sequences, we propose a data augmentation strategy to generate a set of anomalous sequences via negative sampling so that we can train a binary classification model based on the observed normal sequences and the generated anomalous sequences. After training, the classification model is able to detect real anomalous sequences. We then consider the anomalous event detection as a model interpretation problem and apply an interpretable machine learning technique in a novel way to detect which parts of the sequences, a.k.a, anomalous events, lead to anomalous issues. Experimental results on three log datasets show the effectiveness of our proposed framework.

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Citation
```
@inproceedings{han2021interpretablesad,
  title={InterpretableSAD: Interpretable Anomaly Detection in Sequential Log Data},
  author={Han, Xiao and Cheng, He and Xu, Depeng and Yuan, Shuhan},
  booktitle={2021 IEEE International Conference on Big Data (Big Data)},
  pages={1183--1192},
  year={2021},
  organization={IEEE}
}
```
