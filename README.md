## Directed Hypergraph Representation Learning for Link Prediction
Created by Zitong Ma, Wenbo Zhao, Zhe Yang from Soochow University.


### Introduction
This work will appear in AISTATS 2024. Specifically, our work can be concluded into two sophisticated aspects: (1) We define the approximate Laplacian of the directed hypergraph, and further formulate the convolution operation on the directed hypergraph structure, solving the issue of the directed hypergraph structure representation learning. (2) By efficiently learning complex information from directed hypergraphs to obtain high-quality representations, we develop a framework DHGNN for link prediction on directed hypergraph structures. We empirically show that the merit of DHGNN lies in its ability to model complex correlations and encode information effectively of directed hypergraphs. Extensive experiments conducted on multi-field datasets demonstrate the superiority of the proposed DHGNN over various state-of-the-art approaches.
### Citation
if you find our work useful in your research, please consider citing:


### Installation
Install [Pytorch 1.5.0](https://pytorch.org/). You also need to install yaml. The code has been tested with Python 3.7, Pytorch 1.5.0 and CUDA 10.2.

### Usage

To train and evaluate DHGNN for node classification:
```
python run.py
```


To change the experimental parameters:
```
utils/parser.py
```
### License
Our code is released under MIT License (see LICENSE file for details).

