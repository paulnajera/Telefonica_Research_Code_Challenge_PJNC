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
1. Download dataset 
2- Split data 
   IID distribution:
      40 attributes information to be split between clients, equally and random way
   non-IID distribution:
      Choose wich attributes and split them between clients (needs work to be done)
3. Call model pre-trained MobileNetV2, freeze al the layers that have the function tag and modify the las layer to accept
   the number of attributes or classes.
4. Design train and evaluate function (evaluate function needs work to be done, parameter to be used does not fit multilabel. 
5. Desgin Federate Learning
6. Train our model in a local way or using Federate Learning way.
  
