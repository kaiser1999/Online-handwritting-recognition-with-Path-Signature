# Online handwritting recognition with path signature

This is an ongoing project in the book [1] which aims to recognize online handwritten Chinese characters with path signature. The dataset used is [CASIA-OLHWDB2.0-2.2]<http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html> [3]. The results are updated regularly in **_Signature_.out_**.

We here adopt mostly similar model structure in [2] except some minor adjustments:
1. The training inputs are fixed at size 126x2400 instead of 126x576, where each training input represents a line of Chinese characters in the paragraph
2. The rotation problem in each training input is corrected by using **_cv2.minAreaRect()_** methods, i.e. use a rectangle to frame all characters and compute rotation angle
3. Instead of the same convolutional blocks before FCRN blocks (namely FCRN, 2C-FCRN, 3C-FCRN), we use three sets of individual convolutional blocks
4. Instead of BatchNormalization being applied after all but the first two convolutional layers, InstanceNormalization is adopted.
5. No padding is applied before each convolutional layers before FCRN blocks 
6. The first BLSTM layer in each FCRN block is not residual, simply because the input and output dimensions are different
7. Adam optimizer with an exponentially decaying learning rate is adopted instead of Adadelta

The spec of this set-up is:
1. tensorflow - 2.13
2. CUDA - 11.7
3. cuda-nvcc - 12.2.91  
4. nvidia-pyindex - 1.0.9
5. 3 Tesla-V100 GPUs to support a batch size of 36

# Reference
[1]: Kaiser Fan, Phillip Yam (Expected 2024) . Statistical Deep Learning with Python and R.

[2]: Xie, Z., Sun, Z., Jin, L., Ni, H., & Lyons, T. (2017). Learning spatial-semantic context with fully convolutional recurrent network for online handwritten chinese text recognition. IEEE transactions on pattern analysis and machine intelligence, 40(8), 1903-1917.

[3]: C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, Online and Offline Handwritten Chinese Character Recognition: Benchmarking on New Databases, Pattern Recognition, 46(1): 155-162, 2013.