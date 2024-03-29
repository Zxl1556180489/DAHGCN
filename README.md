# DAHGCN
This is an official implementation of Dynamical Attention Hypergraph Convolutional Network for Group Activity Recognition. The details of our source code will be available soon.

## Abstract
Recently, group activity recognition has drawn growing interests in video analysis and computer vision communities. The current models of group activity recognition tasks are often impractical in that they suppose that all interactions between actors are pairwise, which only models and leverages part of the information in real entire interactions. Motivated by this, we design a distinct Dynamical Attention HyperGraph Convolutional Network framework, referred to as DAHGCN, for precise group activity recognition, modeling the entire interactions and capturing the high-order relationships among involved actors in a real-life scenario. Specifically, to learn complementary feature representations for fine-grained group activity recognition, a multi-level feature descriptor module is proposed. Furthermore, for learning higher-order interaction relationships, we construct a dynamical attention hypergraph convolutional network to accommodate complex group interactions, which can dynamically change the topology of the hypergraph and learn these key representations by virtue of the “similarity-based shared nearest neighbor clustering” and “attention mechanisms” on hypergraph. Lastly, a multi-scale temporal convolution module is utilized to explore various long-range temporal dynamic correlations across different frames. Additionally, comprehensive experiments on three commonly used GAR datasets clearly demonstrate that, when compared with the state-of-the-art methods, our proposed method can achieve the most optimal performance.

## Dependencies

- Python `3.x`
- PyTorch `0.4.1`
- numpy, pickle, scikit-image
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)
- Datasets: [Volleyball](https://github.com/mostafa-saad/deep-activity-rec), [Collective](http://vhosts.eecs.umich.edu/vision//activity-dataset.html)

## Prepare Datasets

1. Download [volleyball](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) or [collective](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip) dataset file.
2. Unzip the dataset file into `data/volleyball` or `data/collective`.
