# Telefonica_Research_Code_Challenge_PJNC
Telefonica Research Code Challenge Federated Learning 


Summary:

Build a Federated Learning setting using Flower (https://flower.ai) and PyTorch that uses Large-
scale CelebFaces Attributes (CelebA) Dataset.

Method:
1) Design a data splitter to federate the data creating 2 data distributions:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	a. IID distribution\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	b. non-IID distribution
2) Clearly define training and testing datasets, classes, etc.
3) Use a pre-trained version of MobileNetV2; Freeze the feature extractor; Train the classifier
head from scratch
4) Execute the federated learning training with 50 clients for at least 10 FL rounds
5) Report the training and testing performance with appropriately selected metrics
6) Compare performance across demographic groups present in the CelebFaces dataset

Steps completed:
1. Download dataset (more detail in code)
2. Split data 
   -IID distribution:
      40 attributes information to be split between clients, equally and random way (more detail in code)
   -non-IID distribution:
      Choose wich attributes and split them between clients (needs work to be done
3. Call model pre-trained MobileNetV2, freeze al the layers that have the function tag and modify the last layers to accept
   the number of attributes or classes.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I froze the Layers (Feature extrantion ones) from 0 to 18:
\
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (features): Sequential(\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (0): Conv2dNormActivation(\
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2): ReLU6(inplace=True)\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;).......\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18)\
\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;And modify the classifier head to train my model, this in order to change the out featres:\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(classifier): Sequential(\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(0): Dropout(p=0.2, inplace=False)\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1): Linear(in_features=1280, out_features=1000, bias=True)\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)

   
5. Design train and evaluate function (evaluate function needs work to be done, parameter to be used does not fit multilabel.

Due I used a multilabel classification with 40 classes, the most adequate criterion to use as loss function were:\
    criterion = nn.BCEWithLogitsLoss() or \
    criterion = nn.BCELoss() \
a binary classification for every of the classes.

6. Desgin Federate Learning
7. Train our model in a local way or using Federate Learning way.


# Steps to run code:
 #Start of process 
 1. Load data:  \
train_set, test_set, val_set = load_datasets()
2. Split data and crete subsets: \
train_loaders, test_loaders, val_loaders = iid_data_split_w_data_loaders(train_set, test_set, val_set)
3. Define model: \
model = mobilenetv2_model().to(DEVICE)
4. Train and test model: \
&nbsp;&nbsp;-In a local way:\
   &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train_model(model, train_loader, 1)\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;accuracy = test_model(model, val_loader)\
&nbsp;&nbsp;-With Federate learning:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fl.simulation.start_simulation(\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;client_fn=client_fn, \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;num_clients=NUM_CLIENTS,\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  config=fl.server.ServerConfig(num_rounds=10),\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; strategy=strategy,\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;client_resources=client_resources,\
)


# Missing parts:
1. Evaluation in the test, it wa taking me long time to run and modify this part
2. Therefore, the tuning was no possible to make
3. Performance assessment

Versions:
Package || Version:
Python     3.10.14
Pythorch   2.2.2
