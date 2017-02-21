# Simultaneous Generation-Classification Using LSTM

In this repository you can find the implementation of the Neural Network architecture presented in the paper: Simultaneous Generation-Classification Using LSTM, authored by Daniel Marino, Kasun Amarasinghe, Milos Manic, presented at the 2016 IEEE Symposium Series on Computational Intelligence (SSCI). DOI: 10.1109/SSCI.2016.7850115

The idea that a concept is properly learned by an agent only when the agent is able to reproduce examples of it, is a mechanism that humans often use to evaluate their understanding of a particular concept. In this paper, we propose an LSTM-based architecture that was designed for simultaneously:  

1. generating sequences for a given class 

2. given a sequence, predicting the class to which it belongs

The presented generation-classification methodology was implemented on a sentiment analysis task. However, it can be applied to any sequence modelling or classification task. The following are the main advantages of the presented architecture:

* The experimental results suggest that jointly training the network to achieve simultaneous generation-classification improves generalization on the classification task. This approach of using the generation procedure as a regularization technique reassembles the use of Restricted Boltzmann Machines or auto-encoders for pre-training of deep neural networks. 
* The generation of sequences allows to directly prove the network, which allows to understand and evaluate the performance of the network and how it is behaving. 

The following figure illustrates the proposed architecture when used for generating/classifying sequences that belong to two classes (e.g. positive and negative classes):
