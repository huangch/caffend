# CAFFEnd

Updated using the Caffe master (04-Sept-2017)

CAFFEnd is the Caffe with the extension to any dimensional data set. 

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley.

At beginning, caffe is designed for 2D color and gray scale images. It manages the input data in the fashion of (batch, channel, height and width). As a result, it limits the use cases of more than 2D. Although in recently, newer caffe components are designed to be compatible with any dimensional datasets, also, a newer version of NVIDIA cuDNN also partially supports any dimensional dataset. However, caffe itself is not ready yet for any dimensional datasets.

Thus, I decided to put my effort into making the caffe compatible to any dimensional datasets for my personal interests.
