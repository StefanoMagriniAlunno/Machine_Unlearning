import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

N_CLASSES = 10


class Classifier(torch.nn.Module):
    """This classifier classify images of 32x32 pixels and 3 channel"""

    CNN: torch.nn.Sequential

    class InceptionConv2d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.5):
            super(Classifier.InceptionConv2d, self).__init__()

            out_channels_momentum = int(out_channels * momentum)

            if out_channels <= out_channels_momentum or out_channels_momentum <= 0:
                raise ValueError("momentum value uncorrect")

            self.encode = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(in_channels, out_channels_momentum, (1, 1)),
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels - out_channels_momentum,
                        (3, 3),
                        padding=(1, 1),
                    ),
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y1 = self.encode[0](x)  # 1x1
            y3 = self.encode[1](x)  # 3x3
            y = torch.cat([y1, y3], dim=1)
            return y

    class CapacitorLogical(torch.nn.Module):
        def __init__(self, in_features: int, deep: int, size: int):
            super(Classifier.CapacitorLogical, self).__init__()

            if deep < 0:
                raise ValueError("deep must be greater than 0")
            if size < 1:
                raise ValueError("size must be greater than 0")

            size_ = size * in_features

            self.encode = torch.nn.Linear(in_features, size_)
            self.relu = torch.nn.ReLU(size_)
            self.dropout = torch.nn.Dropout(0.1)
            self.list = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(size_, size_),
                        torch.nn.PReLU(size_),
                    )
                    for _ in range(deep)
                ]
            )
            self.decode = torch.nn.Linear(size_, in_features)
            self.bn = torch.nn.BatchNorm1d(in_features)
            # relu
            # dropout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.encode(x)
            x = self.relu(x)
            x = self.dropout(x)
            for layer in self.list:
                x = layer(x)
            x = self.decode(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x

    def __init__(self, n_classes: int):
        """Initialize the classifier

        Args:
            n_classes (int): num of classes of the classifier
        """
        super(Classifier, self).__init__()

        # sequence of layers
        self.CNN = torch.nn.Sequential(
            # CNN
            # encrease 2
            Classifier.InceptionConv2d(3, 48),
            torch.nn.PReLU(48),
            Classifier.InceptionConv2d(48, 48),
            torch.nn.PReLU(48),
            Classifier.InceptionConv2d(48, 48),
            torch.nn.PReLU(48),
            Classifier.InceptionConv2d(48, 48),
            torch.nn.BatchNorm2d(48),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.SiLU(),
            Classifier.InceptionConv2d(48, 96),
            torch.nn.PReLU(96),
            Classifier.InceptionConv2d(96, 96),
            torch.nn.PReLU(96),
            Classifier.InceptionConv2d(96, 96),
            torch.nn.PReLU(96),
            Classifier.InceptionConv2d(96, 96),
            torch.nn.BatchNorm2d(96),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.SiLU(),
            Classifier.InceptionConv2d(96, 192),
            torch.nn.PReLU(192),
            Classifier.InceptionConv2d(192, 192),
            torch.nn.PReLU(192),
            Classifier.InceptionConv2d(192, 192),
            torch.nn.PReLU(192),
            Classifier.InceptionConv2d(192, 192),
            torch.nn.BatchNorm2d(192),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.SiLU(),
            # tail
            torch.nn.Flatten(),
            torch.nn.Linear(3_072, 1_024),
            torch.nn.RReLU(),
        )
        self.MLP = torch.nn.Sequential(
            # MLP
            Classifier.CapacitorLogical(1_024, 2, 2),
            torch.nn.Linear(1_024, 512),
            torch.nn.PReLU(512),
            Classifier.CapacitorLogical(512, 8, 2),
            # tail
            torch.nn.Linear(512, 256, bias=False),
            torch.nn.BatchNorm1d(256),
            torch.nn.SiLU(),
            # LOGITS
            torch.nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.CNN(x)
        x = self.MLP(x)
        return x


class FilterSet(Dataset):
    """Filter the dataset to have only the classes in targets."""

    dataset: CIFAR10
    access: torch.Tensor

    def __init__(self, dataset: CIFAR10, targets: torch.Tensor):
        self.dataset = dataset
        self.access = torch.nonzero(
            torch.isin(torch.tensor(dataset.targets), targets)
        ).flatten()

    def __len__(self) -> int:
        return self.access.numel()

    def __getitem__(self, idx: int):
        if idx >= self.__len__():
            raise ValueError("idx must be less than {}".format(self.__len__()))
        return self.dataset[self.access[idx]]


def main(n_classes: int, epochs: int, batch_size: int, save_dir: str):
    """Train a classifier with n_classes classes

    Args:
        n_classes (int): num of classes of the classifier
        epochs (int): num of epochs to train the classifier
        batch_size (int): batch size of the training
        save_dir (str): path where save the model
    """
    import os
    import random
    from typing import List

    import torch.utils.data
    from common.pretty import classification_test, classification_train
    from torchvision import transforms

    class CombineOptim(torch.optim.Optimizer):

        list: List[torch.optim.Optimizer]

        def __init__(self, optimizer: torch.optim.Optimizer, *args):
            self.list = [optimizer]
            for arg in args:
                self.list.append(arg)

        def step(self, closure=None):
            return [opt.step(closure) for opt in self.list]

        def zero_grad(self):
            for opt in self.list:
                opt.zero_grad()

    # filter the dataset
    # choose n_classes from the N_CLASSES classes
    targets: List[int] = random.sample(range(N_CLASSES), n_classes)
    dataset = FilterSet(
        CIFAR10(
            root="data/db",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    transforms.RandomAffine(degrees=0, scale=(1.0, 1.1), shear=0),
                    transforms.ColorJitter(
                        contrast=(0.9, 1.5),
                        saturation=(0.9, 1.3),
                        brightness=(0.9, 1.3),
                        hue=(-0.05, 0.05),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            ),
            target_transform=lambda x: targets.index(x),
        ),
        torch.tensor(targets),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = Classifier(n_classes)

    optimizer_CNN = torch.optim.Adam(model.CNN.parameters(), lr=0.0001)
    optimizer_MLP = torch.optim.SGD(model.MLP.parameters(), lr=0.01, momentum=0.9)

    optimizer = CombineOptim(optimizer_CNN, optimizer_MLP)

    device = torch.device("cuda")

    # definisco la regolarizzazione L1 per il modello
    def regularization(model: Classifier) -> torch.Tensor:
        x = torch.cat([p.view(-1) for p in model.CNN.parameters()])
        return 0.0001 * x.norm(1) / x.numel()

    classification_train(
        n_classes,
        model,
        loader,
        optimizer,
        epochs,
        device,
        regularization=regularization,
    )

    test_loader = torch.utils.data.DataLoader(
        FilterSet(
            CIFAR10(
                root="data/db",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                        ),
                    ]
                ),
                target_transform=lambda x: targets.index(x),
            ),
            torch.tensor(targets),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    classification_test(
        n_classes,
        model,
        test_loader,
        device,
    )

    # save the model
    # the name is the list of the classes
    torch.save(
        model.state_dict(), os.path.join(save_dir, "_".join(map(str, targets)) + ".pt")
    )


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    main(8, 30, 256, "data/models/Cifar10")
