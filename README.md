# Hopfield-Neural-Network
This repository contains the implementation of Hopfield Network using Hebbian Learning algorithm. The network is used as auto-associative memory for Image retrieval.

## Implemented Things
* Single Pattern Image retrieval.
* Multiple Pattern Image retrieval.
* Multiple Pattern Image retrieval by making X % of the weight 0.

## Task 1

**Image Visualization**

![](Images/Part_1/Ball.png)  ![](Images/Part_1/Cat.png)  ![](Images/Part_1/Mona.png)

## Task 2
Image of the ball is saved in the network.

**[a] Initialize a zero matrix of the same size as that of the input image of the ball and replace a small patch with a portion of the input image. Use this patch image as the cue for retrieving the image.**

![](Images/Part_1/Ball_patch.png)


**Image Retrieval**

![](Images/Part_2/Iteration_1.png) ![](Images/Part_2/Iteration_2.png) ![](Images/Part_2/Iteration_3.png) ![](Images/Part_2/Iteration_4.png) ![](Images/Part_2/Iteration_5.png) ![](Images/Part_2/Iteration_6.png) ![](Images/Part_2/Iteration_7.png) ![](Images/Part_2/Iteration_8.png) ![](Images/Part_1/Ball.png)

**[b] Plot the Root Mean Squared (RMS) error with time.**

RMS = array([1.26719287, 0.76768049, 0.47702784, 0.2921187 , 0.17888544, 0.11352924, 0.06324555, 0.03651484, 0.00020021])

![](Images/Part_2/RMS_plot.png)

## Task 3
Save all the three images (Ball, Cat, Monalisa) in the network.

**[a] Give small patches of each image to retrieve the corresponding saved image.**

![](Images/Part_1/Cat_patch.png) ![](Images/Part_1/Mona_patch.png)

  
