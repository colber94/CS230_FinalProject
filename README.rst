=============================
Estimating Access to Mobile Broadband 
=============================

.. image:: https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667
        :target: https://colab.research.google.com/github/colber94/CS230_FinalProject/blob/master/colab_UNET.ipynb
        

This is a github repo for a **Stanford CS230** final project. This aim of this project was to take existing data on acutal access to mobile broadband throughout the US and create a Deep Learning model to estimate this using satellite imagery and known locations of cell towers.

This project leverages a CNN model for image segmentation, called **U-Net**. You can find the original paper here
`here. <https://arxiv.org/pdf/1505.04597.pdf>`_. The baseline model we took from the repo used in this paper, `Radio Frequency Interference mitigation using deep convolutional neural networks <http://arxiv.org/abs/1609.09077>`_.

To build and test our model, we used Kaggle, Google Colab, and AWS to leverage GPU's using Tensorflow. 

Much of the work was generating our own data for training. To get the training data, we used satellite images from the Sentinel2, added on a fourth class with cell tower locations and then resized the original 5490x5490 pixel images to 1280x1280 before cropping to 256 x 256. 

|


.. figure:: https://github.com/colber94/CS230_FinalProject/blob/master/images/train.png
   :alt: Segmentation of a galaxies.
   :align: center
   :figclass: align-center
        

|

For the ground truth, we leveraged collected data from the FCC in the form of shapefiles. We performed some data processing to create images with geographic areas broken down by coverage. For our final models we had ground-truth images with a **5 class mask**. 

|
|

.. figure:: https://github.com/colber94/CS230_FinalProject/blob/master/images/truth.png
   :alt: Segmentation of a galaxies.
   :align: center
   
   
After generating the data pipeline, we implemented this on data for six states: AK, AL, CA, CO, MS, OR. 

To 
   
.. figure:: https://github.com/colber94/CS230_FinalProject/blob/master/images/model.png
   :alt: Segmentation of a galaxies.
   :align: center

.. image:: https://github.com/colber94/CS230_FinalProject/blob/master/images/results.png
   :alt: Segmentation of a galaxies.
   :align: center

