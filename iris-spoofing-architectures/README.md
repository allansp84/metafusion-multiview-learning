### CNN architectures for irises spoofing detection

* ***random-filters:*** This directory contains the parameters of the best
CNN architecture found during the architecture optimization process using the algorithm proposed in [1].

* ***spoofnets-cudaconvnets:*** This directory contains the architecture definition of the Spoofnet proposed in [2].
In this case, the filter weights were optimized using the Cuda-convnet2 tool [3].

**References:**

> 1. J. Bergstra, D. Yamins, and D. D. Cox,
"Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures,"
in International Conference on Machine Learning, 2013.
> 2. D. Menotti et al.,
"Deep Representations for Iris, Face, and Fingerprint Spoofing Detection," in IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 864-879, April 2015.
> 3. A. Krizhevsky, "cuda-convnet: High-performance c++/cuda implemen-
tation of convolutional neural networks," 2012.
[Online]. Available: https://github.com/akrizhevsky/cuda-convnet2
