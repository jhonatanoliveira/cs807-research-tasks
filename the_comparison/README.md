# The Comparison

## About

Implementing Sum-product networks (SPNs) shows challenges for a resource constrain computing point of view.
This is due to the size and processing requirements of these models.
SPNs are a layer-wise mathematical models for learning and inference with probabilistic graphical model.
Given a SPN, inference is done by propagating up and down in the network.
Here, we propose one possible implementation for SPNs by using GPUs when parallelizing computation on the up and down in the networks.