from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from numpy import random
import torch
from torch import randperm
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CelebA
import torchvision.models as models
import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
import itertools
from typing import Sequence, cast

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
# print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
# disable_progress_bar()

NUM_CLIENTS = 10
BATCH_SIZE = 32
NUM_ROUNDS = 10
NUM_CLASSES = 40
classes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
           'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
           'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
           'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
           'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
           'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def mobilenetv2_model():
    """
    Function to call the pre-trained model mobilenet_v2
    (weights='IMAGENET1K_V1') ==  (pretrained = True),
    :return: model with feature extractor layers frozen and head modified to use the attributes (0-40)
    """
    num_classes = NUM_CLASSES
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    # Freeze feature extractor layers
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # Modify classifier head
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    # or second way; possible option, add extra sigmoid layer if we want to have values btw 0-1, that would be in binary classification
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(1280, num_classes)  # Adjust output units to match the number of classes
    # )
    return model


def load_datasets():
    """
    Function to download Celeba dataset, transform the data to a specific data,
    split in test, validation and test the dataset and in case 3, use a Data Loader right here
    Option 1.1: Download, split manually, 90% training, 10% test. Subsequently, split training data to have
    validation data (90% and 10%). All splits are random.
    Option 1.2: Download, split manually, 90% training, 10% test. All splits are random.
    Option 2: Download and split manually dataset. The dataset Celeba has an argument to select in wthat kind of
    data we want to split
    :return: training, test and validation
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
        ])
    # Download and load CelebA dataset

    # # # Option 1.1
    # celeba_dataset = CelebA(root='./celeba_data', split='all', target_type='attr', transform=transform, download=True)
    # if "" in celeba_dataset.attr_names: celeba_dataset.attr_names.remove("") #this help to just have a tensor of 40 attibutes,
    #
    # # Split CelebA dataset into training and testing sets
    # train_val_size = int(0.8 * len(celeba_dataset))
    # test_size = len(celeba_dataset) - train_val_size
    # train_val_set, test_set = random_split(celeba_dataset, [train_val_size, test_size], torch.Generator().manual_seed(42))
    #
    # train_size = int(0.9 * len(train_val_set))
    # val_size = len(train_val_set) - train_size
    # train_set, val_set = random_split(train_val_set, [train_size, val_size], torch.Generator().manual_seed(42))
    # return train_set, test_set, val_set

    # # Option 1.2, no validation
    # celeba_dataset = CelebA(root='./celeba_data', split='all', target_type='attr', transform=transform, download=True)
    # if "" in celeba_dataset.attr_names: celeba_dataset.attr_names.remove("")
    #
    # # Split CelebA dataset into training and testing sets
    # train_val_size = int(0.8 * len(celeba_dataset))
    # test_size = len(celeba_dataset) - train_val_size
    # train_val_set, test_set = random_split(celeba_dataset, [train_val_size, test_size],
    #                                        torch.Generator().manual_seed(42))
    # val_set = None
    # return train_set, test_set, val_set


    # Option 2
    train_set = CelebA('./celeba_data', split='train', download=True, transform=transform)
    test_set = CelebA('./celeba_data', split='test', download=True, transform=transform)
    val_set = CelebA('./celeba_data', split='valid', download=True, transform=transform)
    if "" in train_set.attr_names: train_set.attr_names.remove("")
    if "" in test_set.attr_names: test_set.attr_names.remove("")
    if "" in val_set.attr_names: val_set.attr_names.remove("")
    return train_set, test_set, val_set


def get_dataloader(dataset, shuffle=True):
    """
    Function to get DataLoaders from datasets with size batch, dataset does not need to be split between clients
    :param dataset: dataset from which to get the data loader)
    :param batch_size: zise of batch
    :param shuffle:
    :return: Subset shuffle of size batch
    """
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def iid_data_split_w_data_loaders(train_set, test_set, val_set):
    """
    Function to split the data between clients in an equal and random way, drop if necessary when data // client is not 0
    And then create subsets with size batch.
    Two options, one when the load dataset use option 1.2, no need of modifications, it is done automatically
    :param train_set:
    :param test_set:
    :param val_set:
    :return: subsets trainloaders, testloader, valloaders
    """
    if val_set == None: #Option 1.2
        # Split training set into 'num_clients' partitions to simulate different local datasets
        # Determine the number of samples per client
        partition_size = len(train_set) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS
        if len(train_set) % NUM_CLIENTS == 0:
            datasets = random_split(train_set, lengths, torch.Generator().manual_seed(42))
        else:
            indices = randperm(sum(lengths), generator=torch.Generator().manual_seed(42)).tolist()  # type: ignore[arg-type, call-overload]
            lengths = cast(Sequence[int], lengths)
            datasets = [Subset(val_set, indices[offset - length: offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths_train_val = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths_train_val, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
        testloader = DataLoader(test_set, batch_size=BATCH_SIZE)

    else:
        # Split training set into 'num_clients' partitions to simulate different local datasets
        # Determine the number of samples per client
        train_partition_size = len(train_set) // NUM_CLIENTS
        train_lengths = [train_partition_size] * NUM_CLIENTS
        if len(train_set) % NUM_CLIENTS == 0:
            train_datasets = random_split(train_set, train_lengths, torch.Generator().manual_seed(42))
        else:
            indices = randperm(sum(train_lengths), generator=torch.Generator().manual_seed(42)).tolist()  # type: ignore[arg-type, call-overload]
            lengths = cast(Sequence[int], train_lengths)
            train_datasets = [Subset(val_set, indices[offset - length: offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]

        val_partition_size = len(val_set) // NUM_CLIENTS
        val_lengths = [val_partition_size] * NUM_CLIENTS
        if len(val_set) % NUM_CLIENTS == 0:
            val_datasets = random_split(val_set, val_lengths, torch.Generator().manual_seed(42))
        else:
            indices = randperm(sum(val_lengths), generator=torch.Generator().manual_seed(42)).tolist()  # type: ignore[arg-type, call-overload]
            lengths = cast(Sequence[int], val_lengths)
            val_datasets = [Subset(val_set, indices[offset - length: offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds_train in train_datasets:
            trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        for ds_val in val_datasets:
            valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True))
        testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    return trainloaders, testloader, valloaders


def iid_data_split(train_set, test_set, val_set):
    """
        Function to split the data between clients in an equal and random way, drop if necessary when data // client is not 0
    :param train_set:
    :param test_set:
    :param val_set:
    :return: datasets train_set_split, test_set, val_set_split
    """
    if val_set == None: #Option 1.2
        # Split training set into 'num_clients' partitions to simulate different local datasets
        # Determine the number of samples per client
        partition_size = len(train_set) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS
        if len(train_set) % NUM_CLIENTS == 0:
            datasets = random_split(train_set, lengths, torch.Generator().manual_seed(42))
        else:
            indices = randperm(sum(lengths), generator=torch.Generator().manual_seed(42)).tolist()  # type: ignore[arg-type, call-overload]
            lengths = cast(Sequence[int], lengths)
            datasets = [Subset(val_set, indices[offset - length: offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]

        # Split each partition into train/val and create DataLoader
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths_train_val = [len_train, len_val]
            train_set_split, val_set_split = random_split(ds, lengths_train_val, torch.Generator().manual_seed(42))

    else:
        # Split training set into 'num_clients' partitions to simulate different local datasets
        # Determine the number of samples per client
        train_partition_size = len(train_set) // NUM_CLIENTS
        train_lengths = [train_partition_size] * NUM_CLIENTS
        if len(train_set) % NUM_CLIENTS == 0:
            train_set_split = random_split(train_set, train_lengths, torch.Generator().manual_seed(42))
        else:
            indices = randperm(sum(train_lengths), generator=torch.Generator().manual_seed(42)).tolist()  # type: ignore[arg-type, call-overload]
            lengths = cast(Sequence[int], train_lengths)
            train_set_split = [Subset(val_set, indices[offset - length: offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]
        #split validation into / clients
        val_partition_size = len(val_set) // NUM_CLIENTS
        val_lengths = [val_partition_size] * NUM_CLIENTS
        if len(val_set) % NUM_CLIENTS == 0:
            val_set_split = random_split(val_set, val_lengths, torch.Generator().manual_seed(42))
        else:
            indices = randperm(sum(val_lengths), generator=torch.Generator().manual_seed(42)).tolist()  # type: ignore[arg-type, call-overload]
            lengths = cast(Sequence[int], val_lengths)
            val_set_split = [Subset(val_set, indices[offset - length: offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]

    return train_set_split, test_set, val_set_split


#function needs work to be done, I implemented at the beginning before the model, therefore it does not have relevance
def non_iid_data_split(data_train, num_clients, num_attributes=None, attributes=None):
    """
    To split dataset or number of classes/tributes in a random way betwee the clients
    :param data_train:
    :param num_clients:
    :param num_attributes:
    :param attributes:
    :return:
    """

    # Attributes
    # 5 landmark locations, 40 binary attributes annotations per image.

    # obtain which attributes to use in a random way and get the names of those attributes
    if num_attributes is None:
        # Obtain position/index of attributes
        attributes_index = []
        [attributes_index.append(data_train.dataset.attr_names.index(attributes[i])) for i in range(len(attributes))]
        attributes_index = sorted(attributes_index)
        num_attributes = len(attributes_index)
    else:
        # Obtain which attributes to use
        attributes_index = sorted(random.randint(0, 39, size=num_attributes))
        attributes_name = []
        for i in range(num_attributes):
            attributes_name.append(data_train.dataset.attr_names[attributes_index[i]])

    # Create group of clients
    clients_groups = (np.array_split(range(num_clients), 2**num_attributes))
    # dict for saving the values of index for every class or attributes
    dict_classes_index = {}
    # dic of list of diff ways of combinations
    dict_values_classes = {}
    list_of_combinations = list(itertools.product([0, 1], repeat=num_attributes))
    for i in range(2**num_attributes):
        dict_classes_index[i] = []
        dict_values_classes[i] = list(list_of_combinations[i])

    # go through every index's tensor and check if its values fits in any of those classes previously decided
    for i in data_train.indices:
        attributes_values = list(np.array(data_train.dataset.attr.data[i])[attributes_index])
        for j in range(2**num_attributes):
            if attributes_values == dict_values_classes[j]:
                dict_classes_index[j].append(i)
                break

    dict_clients = {}
    for u, group in enumerate(clients_groups):
        # Determine the number of samples per client
        samples_per_client = len(dict_classes_index[u]) // len(clients_groups[u])
        # Split the data classes into equal portions for each group of client
        client_data_indices = [dict_classes_index[u][i:(i + samples_per_client)] for i in
                               range(0, samples_per_client * len(group), samples_per_client)]
        for a, b in enumerate(group):
            dict_clients[b] = Subset(data_train, client_data_indices[a])

    return dict_clients

    # Note: if we increase attributes, the data turns to be more specific, and the number of matches of those attributes
    # could not be enough to split it between clients, or they will receive the same info, i.e. 4 values who match the selected
    # attributes, and we have 10 clients. On the other hand, less selected att. 20000 values and 10 clients


def obtain_classes(batch):
    """
    Function to obtain the name of the attributes who have a value of 1
    :param batch:
    :return:
    """
    positions_batch = []
    for list_num, i in enumerate(batch[1]):
        positions_unit = []
        for pos, j in enumerate(i):
            if j == 1:
                positions_unit.append(classes[pos])
        positions_batch.append(positions_unit)
    return positions_batch


def train_model(model, train_loader, epochs=10):
    """
    Function to train our model
    :param model: mobilenetV2
    :param train_loader: dataset of info to be use to train our model
    :param epochs:
    :return:
    """
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        # print('epoch: ', epoch)
        correct, total, epoch_loss = 0, 0, 0.0
        # for images, labels in train_loader:
        for batch in train_loader:
            # images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            # I need one loss that gives me 1 or 0 or btw 1 and 0 in the same tensor form that label or post processing, numbers close to 1 or 0 turn to be 1 or 0
            loss = criterion(outputs.to(torch.float), labels.to(torch.float))
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)  # Accumulate loss
        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss
        print('Epoch: ', (epoch+1)/(epochs), 'Loss: ', epoch_loss)


def test_model(model, test_loader):
    """
    Fucntion use to test out model, needs to be worked, multilabel criterion gives me thigns to read and how
    to evaluate in a correct matter
    :param model:
    :param test_loader:
    :return:
    """

    """Evaluate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs.to(torch.float), labels.to(torch.float)).item()
            _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # loss /= len(test_loader.dataset)
    # accuracy = correct / total
    # return loss, accuracy


#Implementing Flower
#Functions and parameters to train our model in the clients side
def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

#Implementing a Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_model(self.model, self.train_loader, epochs=1)
        return get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test_model(self.model, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    """
    Funtion to create a Flower client representing a single organization
    :param cid:
    :return:
    """
    # Load model
    model = mobilenetv2_model().to(DEVICE)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(model, train_loader, val_loader).to_client()


# Start of process
 #Load data
train_set, test_set, val_set = load_datasets() #just loaded, not random shuffle

#Block of data not splitted btw clients and extracted by DataLoader
# train_loader = get_dataloader(train_set, shuffle=True)
# test_loader = get_dataloader(train_set, shuffle=True)
# val_loader = get_dataloader(val_set, shuffle=False)

# IID data split
# train_set_split_idd, test_set_split_idd, val_set_split_idd = iid_data_split(train_set, test_set, val_set)  #data split rando btw clients
train_loaders, test_loaders, val_loaders = iid_data_split_w_data_loaders(train_set, test_set, val_set) #data split rando btw clients equally and extracted by DataLoader

# Define model
model = mobilenetv2_model().to(DEVICE)


#Train the model in a local way

# train_model(model, train_loader)
# test_model(model, test_loader)

#with just a set of clients
train_loader = train_loaders[0]
val_loader = val_loaders[0]
for epoch in range(5):
    train_model(model, train_loader, 1)
    # loss, accuracy = test_model(model, val_loader)
    # print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
# loss, accuracy = test_model(model, test_loaders)
# print(f"Final test set performance:\n\t loss {loss}\n\t accuracy {accuracy}")
#
#Train with the whole set of clients and data
# for epoch in range(5):
#     for u in range(NUM_CLIENTS):
#         train_model(model, train_loaders[u], 1)
#         loss, accuracy = test_model(model, val_loaders[u])
#         print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

# loss, accuracy = test_model(model, test_loaders)
# print(f"Final test set performance:\n\t loss {loss}\n\t accuracy {accuracy}")


# torch.save(model.state_dict(), './celeba_data')
# torch.save(model, './celeba_data')


# FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

# Specification of resources that each client need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),  #ten rouds
    strategy=strategy,
    client_resources=client_resources,
)





