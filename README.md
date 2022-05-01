
# Can Abnormality be Detected by Graph Neural Networks?

## AMNet

![AMNet](https://github.com/godcherry/AMNet/blob/main/AMNet.svg)

## Model Usage

### Dependencies 

This project is tested on cuda 11.2 with several dependencies listed below:

```markdown
pytorch=1.11.0
torch-geometric=2.0.4
```

### Input Format

The input graph data is expected to be a `torch_geometric.data.Data` . We organize the preprocessed `Yelp / Elliptic` datasets in the `/dataset`. 

### Main Script 

```markdown
python train.py --dataset yelp/elliptic
```

Tuned dataset-dependent hyper-parameters could be found in `config.py`

```markdown
K: Number of filters
M: Order of Bernstein Polynomials
lr_f: Learning rate of filter module
lr: Learning rate of non-filter module
beta: Weight of marginal constraint loss
```

## Reference
[1] Chai, Z; You, S; Yang, Y; Pu, S; Xu, J; Cai, H and Jiang, W, 2021, Can Abnormality be Detected by Graph Neural Networks?, In IJCAI, 2022

```
@inproceedings{chai2022can,
  title = "{Can Abnormality be Detected by Graph Neural Networks?}", 
  author = {{Chai}, Z. and {You}, S. and {Yang}, Y. and {Pu}, S. and {Xu}, J. and {Cai}, H and {Jiang}, W.}, 
  booktitle={Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI)},
  year = 2022, 
} 
```
