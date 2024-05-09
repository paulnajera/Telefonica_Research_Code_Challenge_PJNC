from collections import OrderedDict
from typing import List, Tuple

import numpy as np
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
    #
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(1280, num_classes)  # Adjust output units to match the number of classes
    # )
    return model


def load_datasets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
        ])
    # Download and load CelebA dataset

    # # # Option 1
    # celeba_dataset = CelebA(root='./celeba_data', split='all', target_type='attr', transform=transform, download=True)
    # if "" in celeba_dataset.attr_names: celeba_dataset.attr_names.remove("")
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

    # # Option 3
    # trainset = CelebA('./celeba_data', split='train', download=True, transform=transform)
    # testset = CelebA('./celeba_data', split='test', download=True, transform=transform)
    # if "" in trainset.attr_names: trainset.attr_names.remove("")
    # if "" in testset.attr_names: testset.attr_names.remove("")
    #
    # # Split training set into `num_clients` partitions to simulate different local datasets, se usa IDD
    # partition_size = len(trainset) // NUM_CLIENTS
    # lengths = [partition_size] * NUM_CLIENTS # [num_of_elements_per_client, num_of_elements_per_client...]
    # datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    #
    # # Split each partition into train/val and create DataLoader
    # trainloaders = []
    # valloaders = []
    # for ds in datasets:
    #     len_val = len(ds) // 10  # 10 % validation set
    #     len_train = len(ds) - len_val
    #     lengths_train_val = [len_train, len_val]
    #     ds_train, ds_val = random_split(ds, lengths_train_val, torch.Generator().manual_seed(42))
    #     trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
    #     valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    # testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    #
    # return trainloaders, valloaders, testloader


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def iid_data_split_w_data_loaders(train_set, test_set, val_set):
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


# def non_iid_data_split(train_set, test_set, val_set):
#     num_samples_per_client = len(dataset) // NUM_CLIENTS
#     indices = torch.randperm(len(dataset)).tolist()
#     client_data = [torch.utils.data.Subset(dataset, indices[i*num_samples_per_client:(i+1)*num_samples_per_client]) for i in range(num_clients)]
#     return client_data


def obtain_classes(batch):
    positions_batch = []
    for list_num, i in enumerate(batch[1]):
        positions_unit = []
        for pos, j in enumerate(i):
            if j == 1:
                positions_unit.append(classes[pos])
        positions_batch.append(positions_unit)
    return positions_batch


def train_model(model, train_loader, epochs=10):
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # verbose = False
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    model.to(DEVICE)
    model.train()

    # batch = next(iter(train_loader))
    # images, labels = batch[0], batch[1]

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
    """Evaluate the network on the entire test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs.to(torch.float), labels.to(torch.float)).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy


#Implementing Flower
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
    """Create a Flower client representing a single organization."""
    # Load model
    model = mobilenetv2_model().to(DEVICE)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(model, train_loader, val_loader).to_client()

#Starting the training

""""""


# Start of process
# #Option3
# trainloaders, valloaders, testloader = load_datasets()

#Option 1 and 2
train_set, test_set, val_set = load_datasets() #just loaded, not random shuffle

#Block of data not splitted btw clients and extracted by DataLoader
# train_loader = get_dataloader(train_set, batch_size=32, shuffle=True)
# test_loader = get_dataloader(train_set, batch_size=32, shuffle=True)
# val_loader = get_dataloader(val_set, batch_size=32, shuffle=False)

# IID data split
# train_set_split_idd, test_set_split_idd, val_set_split_idd = iid_data_split(train_set, test_set, val_set)  #data split rando btw clients
train_loaders, test_loaders, val_loaders = iid_data_split_w_data_loaders(train_set, test_set, val_set) #data split rando btw clients equally and extracted by DataLoader

# batch = next(iter(train_loaders[0]))
# images, labels = batch[0], batch[1]

# Define model
model = mobilenetv2_model().to(DEVICE)

# Train the classifier head from scratch

train_loader = train_loaders[0]
val_loader = val_loaders[0]
# train_model(model, train_loader)
# test_model(model, test_loader)
# Train test local
for epoch in range(2): #5
    train_model(model, train_loader, 1)
    loss, accuracy = test_model(model, val_loader)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
#
# loss, accuracy = test_model(model, test_loaders)
# print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
#

# for epoch in range(5):
#     for u in range(NUM_CLIENTS):
#         train_model(model, train_loaders[u], 1)
#         loss, accuracy = test_model(model, val_loaders[u])
#         print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
#
# loss, accuracy = test_model(model, test_loaders)
# print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")


# torch.save(model.state_dict(), './celeba_data')
# torch.save(model, './celeba_data')

test = 5

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)



""""""




