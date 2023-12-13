import torch
from torch import nn
from torch import optim
import numpy as np
import random
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
import pandas as pd
import time

# create net


class ConvNet(nn.Module):
    def __init__(self, device, weight_init="xavier", num_filters=(16, 32)):
        super(ConvNet, self).__init__()
        self.device = device
        self.weight_init = weight_init
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Sequential(nn.Linear(num_filters[1] * 7 * 7, 10))  
        #cross entropy loss does softmax implicitly
        self.apply(self.init_wieghts)

    def init_wieghts(self, w):
        if isinstance(w, nn.Conv2d) or isinstance(w, nn.Linear):
            torch.nn.init.normal_(w.weight, 0, 0.01)
            if self.weight_init == "xavier":
                torch.nn.init.xavier_normal_(
                    w.weight, gain=nn.init.calculate_gain("sigmoid")
                )
            if self.weight_init == "he":
                torch.nn.init.kaiming_normal_(w.weight, nonlinearity="sigmoid")
            if self.weight_init == "normal":
                torch.nn.init.normal_(w.weight, 0, 0.01)

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.to(self.device)
        else:
            x = torch.from_numpy(x).to(self.device)
        x.requires_grad_()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def get_mnist():
    from mnist_loader import load_data_wrapper

    train, val, test = load_data_wrapper()
    train_x, train_y = zip(*train)
    val_x, val_y = zip(*val)
    test_x, test_y = zip(*test)
    train_x_, val_x_, test_x_ = [], [], []

    for t in train_x:
        train_x_.append(t.reshape((1, 28, 28)))
    for v, ts in zip(val_x, test_x):
        val_x_.append(v.reshape((1, 28, 28)))
        test_x_.append(ts.reshape((1, 28, 28)))

    train_x_, val_x_, test_x_ = np.array(train_x_), np.array(val_x_), np.array(test_x_)

    train_y_ = np.array([np.argmax(np.squeeze(t), axis=0) for t in train_y])
    val_y_ = np.array(val_y)
    test_y_ = np.array(test_y)

    return (train_x_, train_y_), (val_x_, val_y_), (test_x_, test_y_)


class CustomMnistDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        super().__init__()
        self.X = np.divide(X, 255.0)
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x = self.X[index]
        x = Image.fromarray(x, mode="L")
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
            x = torch.permute(x, (1, 0, 2))
        return x, y


def train_epoch(model, train_loader, optimizer, loss_fn, epoch_index, device):
    loss = 0
    for i, batch in enumerate(train_loader):
        x, y = batch
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    print(f"Training Loss for last batch of epoch {epoch_index}: {loss}")
    return loss


from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
    auc,
)


# training code
def train_model(model_num, 
    hyper_params={
        "EPOCHS": 10,
        "OPTIMIZER": "adam",
        "NUM_FILTERS": (32, 64),
        "WEIGHT_INIT": "normal",
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 0.001,
    }
):
    # hyperparams
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    model = ConvNet(
        device=device,
        weight_init=hyper_params["WEIGHT_INIT"],
        num_filters=hyper_params["NUM_FILTERS"],
    )
    model.to(device)
    model_directory = "./results/mnist_models/"

    # create loss func and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params["LEARNING_RATE"])

    """train, (val_x, val_y), _ = get_mnist()
    val_y = torch.from_numpy(val_y).to(device)
    train_dataloader = DataLoader(CustomMnistDataset(train[0],train[1], transform=ToTensor()),batch_size=hyper_params["BATCH_SIZE"],shuffle=True)
    """

    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=ToTensor(),
        download=True,
    )
    train_data, val_data = random_split(train_data, lengths=(5 / 6, 1 / 6))
    val_loader = DataLoader(val_data, batch_size=len(val_data))
    val_x, val_y = next(iter(val_loader))
    val_y = val_y.to(device)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=hyper_params["BATCH_SIZE"], shuffle=True
    )

    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "v_accuracy": [],
        "v_precision": [],
        "v_recall": [],
        "v_f1": [],
        "params": [],
        "model_num": [],
    }
    for EPOCH in range(hyper_params["EPOCHS"]):
        print(f"EPOCH #{EPOCH}")
        model.train()
        train_loss = (
            train_epoch(model, train_dataloader, optimizer, loss_fn, EPOCH, device)
            / hyper_params["BATCH_SIZE"]
        )
        model.eval()

        with torch.no_grad():
            val_y_pred = model(val_x)
            metrics["epoch"].append(EPOCH)
            metrics['model_num'].append(model_num)
            metrics["val_loss"].append(loss_fn(val_y_pred, val_y).item() / len(val_y))
            metrics["train_loss"].append(train_loss.item() / hyper_params["BATCH_SIZE"])
            metrics["v_accuracy"].append(
                multiclass_accuracy(val_y_pred, val_y, num_classes=10).item()
            )
            metrics["v_f1"].append(
                multiclass_f1_score(val_y_pred, val_y, num_classes=10).item()
            )
            metrics["v_precision"].append(
                multiclass_precision(val_y_pred, val_y, num_classes=10).item()
            )
            metrics["v_recall"].append(
                multiclass_recall(val_y_pred, val_y, num_classes=10).item()
            )
            metrics["params"].append(str(hyper_params))
            # v_auc = auc(val_y_pred, val_y)
            # if EPOCH == 10:
            # print(val_y[50:70])
            # print(np.array([np.argmax(np.squeeze(t), axis=0) for t in val_y_pred.cpu()])[50:70])
            print(
                f'Validation Acc: {metrics["v_accuracy"][-1]} | Loss: {metrics["val_loss"][-1]}'
            )
            print( model_directory + f'model_{model_num}_'+ f"epoch_{EPOCH}.pt" )
            torch.save(model.state_dict(), model_directory + f'model_{model_num}'+ f"epoch_{EPOCH}.pt" )

    print(metrics)
    df = pd.DataFrame(metrics)
    return df
    # df.to_csv("./results/mnist.csv")
    #
    # with open("./results/mnist_params.json", "w") as f:
    #     json.dump(hyper_params, f)


def grid_search(
    hyper_param_list={
        "NUM_FILTERS": [(32, 64), (64, 64), (32, 32)],
        "BATCH_SIZE": [8, 32],
        "LEARNING_RATE": [0.001, 0.01],
    }
):
    data = pd.DataFrame()
    count = 1
    
    print(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    for nf in hyper_param_list["NUM_FILTERS"]:
        for bs in hyper_param_list["BATCH_SIZE"]:
            for lr in hyper_param_list["LEARNING_RATE"]:
                time_1 = time.perf_counter()
                hyper_params = {
                    "EPOCHS": 20,
                    "OPTIMIZER": "adam",
                    "NUM_FILTERS": nf,
                    "WEIGHT_INIT": "normal",
                    "BATCH_SIZE": bs,
                    "LEARNING_RATE": lr,
                }
                print("Beginning search")
                print(f"Searching over: {hyper_params}")

                search = train_model(count, hyper_params=hyper_params)
                #search = search[search["epoch"] == 9]
                data = pd.concat([data, search])
                print(count)
                count +=1
                print(f'Time taken for model {count}: {time.perf_counter() - time_1}')

                


    data = data.reset_index()
    #data = data.drop(["index", "epoch"], axis=1)
    data.to_csv("./results/mnist_search.csv")
    return data


if __name__ == "__main__":
    grid_search()
