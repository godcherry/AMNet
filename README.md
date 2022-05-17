
# Can Abnormality be Detected by Graph Neural Networks?

## AMNet

This project implements the Adaptive Multi-frequency filtering graph neural network for graph anomaly detection(AMNet) proposed in [1], which focuses on graph anomaly detection with GNNs exploiting both low/high-frequency signals. 


## Model Usage

### Dependencies 

This project is tested on cuda 11.2 with several dependencies listed below:

```markdown
pytorch=1.11.0
torch-geometric=2.0.4
```

### Input Format

### Dataset 

Two benchmark datasets `Elliptic` and `Yelp` of graph anomaly detection are available for evaluation. `Elliptic` was first proposed in [this paper](https://arxiv.org/pdf/2008.08692.pdf), of which goal is to detect money-laundering users in bitcoin network.
We follow the original paper to perform a temporal split of training, validation and testing data. Note that we take a homophilous version of `Yelp` because we focus on homophilous graph to study the spectral properties of GNNs.

Datasets are processed to be a `torch_geometric.data.Data` . We also provide preprocessed `Yelp / Elliptic` datasets in the `/dataset`. 

### Get start

Train the model on benchmark datasets with the default setting:

```markdown
python train.py --dataset yelp/elliptic
```

Tuned hyper-parameters could be found in `config.py`

### Run on your own dataset

You could organize your dataset into a `torch_geometric.data.Data` then add profile of your own dataset on `config.py`

```markdown
dataset_config = {
    ...
    'custom_dataset':
        {
            'K': 2,  # Number of filters
            'M': 5,  # Order of Bernstein Polynomials
            'hidden_channels': 64, 
            'lr_f': 5e-2, # Learning rate of filter module
            'lr': 5e-4, # Learning rate of non-filter module
            'weight_decay': 1e-5,
            'beta': 1., # Weight of marginal constraint loss
            'epochs': 2000, # total training epoch number
            'patience': 200 # patience for early-stopping
        },
    ...
}
```
Then train your own dataset with the customized setting:

```markdown
python train.py --dataset custom_dataset
```

## Reference

If you think this work is helpful for your project, please give it a star and citation:

```
@inproceedings{chai2022can,
  title = "{Can Abnormality be Detected by Graph Neural Networks?}", 
  author = {Ziwei Chai and Siqi You and Yang Yang and Shiliang Pu and Jiarong Xu and Haoyang Cai and Weihao Jiang, 
  booktitle={Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI)},
  year = 2022, 
} 
```

[1] Chai, Z; You, S; Yang, Y; Pu, S; Xu, J; Cai, H and Jiang, W, 2021, Can Abnormality be Detected by Graph Neural Networks?, In IJCAI, 2022

## Contact

If you have any problem, feel free to contact with me. Thanks.

E-mail: zwchai@zju.edu.cn

## Abstract
Anomaly detection in graphs has attracted considerable interests in both academia and industry due
to its wide applications in numerous domains ranging from finance to biology. Meanwhile, graph neural networks (GNNs) is emerging as a powerful tool
for modeling graph data. A natural and fundamental question that arises here is: can abnormality be
detected by graph neural networks? In this paper,
we aim to answer this question, which is nontrivial.
As many existing works have explored, graph neural networks can be seen as filters for graph signals,
with the favor of low frequency in graphs. In other
words, GNN will smooth the signals of adjacent
nodes. However, abnormality in a graph intuitively
has the characteristic that it tends to be dissimilar to
its neighbors, which are mostly normal samples. It
thereby conflicts with the general assumption with
traditional GNNs. To solve this, we propose a novel
Adaptive Multi-frequency Graph Neural Network
(AMNet)
, aiming to capture both low-frequency
and high-frequency signals, and adaptively combine signals of different frequencies. Experimental
results on real-world datasets demonstrate that our
model achieves a significant improvement comparing with several state-of-the-art baseline methods.


![AMNet](https://github.com/godcherry/AMNet/blob/main/AMNet.gif)