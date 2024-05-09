# Telefonica_Research_Code_Challenge_PJNC
Telefonica Research Code Challenge Federated Learning 


Summary:

Build a Federated Learning setting using Flower (https://flower.ai) and PyTorch that uses Large-
scale CelebFaces Attributes (CelebA) Dataset.

Method:
1) Design a data splitter to federate the data creating 2 data distributions:
	a. IID distribution
	b. non-IID distribution
2) Clearly define training and testing datasets, classes, etc.
3) Use a pre-trained version of MobileNetV2; Freeze the feature extractor; Train the classifier
head from scratch
4) Execute the federated learning training with 50 clients for at least 10 FL rounds
5) Report the training and testing performance with appropriately selected metrics
6) Compare performance across demographic groups present in the CelebFaces dataset

Steps completed:
1. Download dataset (more detail in code)
2- Split data 
   -IID distribution:
      40 attributes information to be split between clients, equally and random way (more detail in code)
   -non-IID distribution:
      Choose wich attributes and split them between clients (needs work to be done
3. Call model pre-trained MobileNetV2, freeze al the layers that have the function tag and modify the last layers to accept
   the number of attributes or classes.

I froze the Layers (Feature extrantion ones) from 0 to 18:
\
  (features): Sequential(\
    (0): Conv2dNormActivation(\
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
      (2): ReLU6(inplace=True)\
    ).......\
    (18)\
\
   And modify the classifier head to train my model, this in order to change the out featres:\
   \
  (classifier): Sequential(\
    (0): Dropout(p=0.2, inplace=False)\
    (1): Linear(in_features=1280, out_features=1000, bias=True)\
  )\
\
   
5. Design train and evaluate function (evaluate function needs work to be done, parameter to be used does not fit multilabel.



6. Desgin Federate Learning
7. Train our model in a local way or using Federate Learning way.





Versions:
Package || Version:
Python     3.10.14
Pythorch   2.2.2
