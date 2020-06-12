=============================
Estimating Access to Mobile Broadband 
=============================

.. image:: https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667
        :target: https://colab.research.google.com/github/colber94/CS230_FinalProject/blob/master/colab_UNET.ipynb
        

Background
########

This is a github repo for a **Stanford CS230** final project. This aim of this project was to take existing data on acutal access to mobile broadband throughout the US and create a Deep Learning model to estimate this using satellite imagery and known locations of cell towers.

This project leverages a CNN model for image segmentation, called **U-Net**. You can find the original paper here
`here <https://arxiv.org/pdf/1505.04597.pdf>`_. The baseline model we took from the repo used in this paper, `Radio Frequency Interference mitigation using deep convolutional neural networks <http://arxiv.org/abs/1609.09077>`_.

To build and test our model, we used Kaggle, Google Colab, and AWS to leverage GPU's using Tensorflow. 

Data Pipeline
########

Much of the work was generating our own data for training. To get the training data, we used satellite images from the Sentinel2, added on a fourth class with cell tower locations and then resized the original 5490 x 5490 pixel images to 1280 x 1280 before cropping to 256 x 256. 

|


.. figure:: https://github.com/colber94/CS230_FinalProject/blob/master/images/train.png
   :alt: TRAINING images
   :align: center
   :figclass: align-center
        

|

For the ground truth, we leveraged collected data from the FCC in the form of shapefiles. We performed some data processing to create images with geographic areas broken down by coverage. For our final models we had ground-truth images with a **5 class mask**. 

|


.. figure:: https://github.com/colber94/CS230_FinalProject/blob/master/images/truth.png
   :alt: truth
   :align: center
   
|   

After generating the data pipeline, we implemented this on data for six states: AK, AL, CA, CO, MS, OR. 

Model Implementation
########

The U-Net model is a CNN that is specifically used for image segementation tasks. It is broken into two parts: the downsampling and upsampling. The downsampling goes through various convultional blocks that include a convolution, dropout, a ReLU activation, and finally a max pool. THe upsampling go through a similar process that includes a convolution and  activation. 

To train our model we started with the base implementation. To train the best model we experimented with many different paramters and hyperparameters. For our data, we experimented with the number of classes (2,4,5,10), the downsizing of the image (5490 to a variety of sizes (1280,2560, etc.)), and the size of the cropping (256, 512 etc.). A problem we ran into was the limited RAM even using AWS or Kaggle, which limited the size and resolution of the images. We also experimented with data augmentation techniques: flipping, rotating, etc., normalizing the input images. 

|
   
.. figure:: https://github.com/colber94/CS230_FinalProject/blob/master/images/model.png
   :alt: Segmentation of a galaxies.
   :align: center
   
|

For the model itself, we experimented with the learning rate, batch size, dropout, layer depth, number of root layers, and finally batch normalization. The original U-Net implementation did not have batch normalization, and upon implementing we saw an **increase in training rate** and **decrease in variance**. 

Results and Future Work
########

We were able to achieve a 97% categorical accuracy on the training set with almost 80% on the test set. 

|
   
.. image:: https://github.com/colber94/CS230_FinalProject/blob/master/images/results.png
   :alt: Segmentation of a galaxies.
   :align: center

|

Future work will include adding more data to decrease variance, training on the uncropped images, and possibly adding in extra layers into the U-Net model itself.

We hope this can be very beneficial for helping to estimate access to mobile broadband and can be used by government institutions and ISP for infrastructure planning and policy-making.


Please checkout the `Youtube Presentation <https://www.youtube.com/watch?v=eY6-gHf1iaQ&lc=Ugxb0CgbtMGqFKvdfjd4AaABAg>`_.
