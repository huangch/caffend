# CAFFEnd
History:

3-OCT-2017: Fixed bugs.

4-SEPT-2017: Updated using the Caffe master.

CAFFEnd is the Caffe with the extension to any dimensional data set. 

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley.

At beginning, caffe was designed for 2D color and gray scale images. It manages the input data in the fashion of (batch, channel, height and width). As a result, it limits the use cases of more than 2D. Although in recently, newer caffe components were designed to be compatible with datasets of any dimensions, also, a newer version of NVIDIA cuDNN also partially supports datasets of any dimensions. However, caffe itself is not ready yet for a dataset which consists of more than 2 or 3 dimensions.

Thus, I decided to put my effort into making the caffe compatible to any dimensional datasets for my personal interests.

VNet, proposed by faustomilletare, is a 3D segmentaion approach (https://github.com/faustomilletari/VNet). This model is a good example for validating the proposed CAFFEnd.  
