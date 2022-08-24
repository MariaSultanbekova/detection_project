## Hi! 
This time I decided to check how the neural network from the mmdetection library can cope with our favorite task of recognizing cats and dogs))

Code https://github.com/MariaSultanbekova/detection_project/blob/main/cat-dog_detection.ipynb

--------------------------------------------------------------------------------------------------------------------------------------------------------
So, let's talk about detection.

The task of detection is to find an object in the image, frame it and sign it. 
We will use the Faster-RCNN model designed for two-stage detection. The idea is that we find an interesting area in the picture, and then determine what is there.

![header](https://github.com/MariaSultanbekova/detection_project/blob/main/detection_image.png)

## Model architecture

![header](https://github.com/MariaSultanbekova/detection_project/blob/main/Region%2BProposal%2BNetwork.jpg)

The architecture is a composition of two modules: Region Proposal Network algorithm and a Fast R-CNN. 

RPN is a sequence of convolutions that search for areas with objects in the image. Then the Roi Pooling algorithm comes into play: we increase the found area and explore separately. We cut out the bounding box found from the tensor, apply bilinear interpolation (change the size for convolution), convolution and predict the coordinates of the object.

![header](https://github.com/MariaSultanbekova/detection_project/blob/main/roi-pooling.png)

# Model Training

So, to train a model from the mmdetection library, we will need to create a COCO format dataset and adjust the config. Then we will configure the parameters and feed the data to the model.

After learning on 3 epochs, let's try to predict a kitty or a dog.

![header](https://github.com/MariaSultanbekova/detection_project/blob/main/prediction_after_train.png)

Hmm, well, even I am tormented by doubts that this is a dog...

What we used here is called transfer learning. We took a model that is already trained to recognize some things and now just fine-tuned it to the task of cats and dogs. That is, the weights we took were not random.

If you train the model on more epochs and experiment with the learning rate, the result will be much better.

--------------------------------------------------------------------------------------------------------------------------------
The beginning was not encouraging...
![header](https://github.com/MariaSultanbekova/detection_project/blob/main/prediction_before_train.png)


