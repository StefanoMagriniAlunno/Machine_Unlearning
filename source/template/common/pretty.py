from typing import Callable, List

import torch
from pandas import DataFrame
from tabulate import tabulate  # type: ignore
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from .classification import batch_loss


def classification_train(
    n_classes: int,
    classifier: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    regularization: Callable[[torch.nn.Module], torch.tensor] | None = None,
):
    torch.cuda.empty_cache()

    classifier.to(device)
    classifier.train()
    print(" Num of parameters = ", sum([p.numel() for p in classifier.parameters()]))
    print("   Num of classes  = ", n_classes)
    print("    Num of data    = ", len(loader.dataset))
    print("    Batch size     = ", loader.batch_size)

    # make the DataFrame with epochs rows
    df = DataFrame(
        columns=["epoch", "accuracy", "confidence", "loss", "l_var", "l_min", "l_max"]
    )
    df["epoch"] = range(epochs)
    df = df.set_index("epoch")

    # loss_fn
    loss_fn = torch.nn.CrossEntropyLoss()

    # train loop
    progress_bar = tqdm(range(epochs), desc="Epochs", leave=False)
    for e in progress_bar:
        # required typing
        loss_list: List[float]
        x: torch.Tensor
        y: torch.Tensor

        # training variables
        accuracy = 0.0
        confidence = 0.0
        loss_list = []

        # internal train loop
        for _, (x, y) in tqdm(
            enumerate(loader), desc="Batch", leave=False, total=len(loader)
        ):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss, att, conf = batch_loss(classifier, x, y, loss_fn, regularization)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            confidence += conf.item()
            accuracy += att.item() / loader.batch_size

        # salvo i risultati nel data frame
        loss_mean = sum(loss_list) / len(loss_list)
        loss_variance = sum([(loss - loss_mean) ** 2 for loss in loss_list]) / len(
            loss_list
        )
        loss_min = min(loss_list)
        loss_max = max(loss_list)
        accuracy /= len(loader)
        confidence /= len(loader)
        df.loc[e, "loss"] = loss_mean
        df.loc[e, "l_var"] = loss_variance
        df.loc[e, "l_min"] = loss_min
        df.loc[e, "l_max"] = loss_max
        df.loc[e, "accuracy"] = accuracy
        df.loc[e, "confidence"] = confidence

        # riporto la loss e l'accuracy nel progress bar
        progress_bar.set_postfix(
            loss=loss_mean,
            accuracy=accuracy,
        )

    print(tabulate(df, headers="keys", tablefmt="pretty"))


def classification_test(
    n_classes: int,
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    torch.cuda.empty_cache()

    classifier.to(device)
    classifier.eval()
    print(" Num of parameters = ", sum(p.numel() for p in classifier.parameters()))
    print("   Num of classes  = ", n_classes)
    print("    Num of data    = ", len(loader.dataset))
    print("    Batch size     = ", loader.batch_size)

    # loss_fn
    loss_fn = torch.nn.CrossEntropyLoss()

    # test variables
    accuracy = 0.0
    confidence = 0.0

    # test loop
    x: torch.Tensor
    y: torch.Tensor
    for x, y in tqdm(loader, desc="Testing", leave=False):
        x, y = x.to(device), y.to(device)

        _, att, conf = batch_loss(classifier, x, y, loss_fn)

        confidence += conf.item()
        accuracy += att.item() / loader.batch_size

    print("Confidence: ", confidence / len(loader))
    print("Accuracy:   ", accuracy / len(loader))

    # calcolo e mostro la matrice di confusione
    with torch.no_grad():
        confusion_matrix = torch.zeros((n_classes, n_classes), device=device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = classifier(x)
            for i in range(len(y)):
                confusion_matrix[y[i], :] += torch.nn.functional.softmax(
                    y_pred[i], dim=0
                )
        # normalizzo la matrice
        confusion_matrix /= torch.sum(confusion_matrix, dim=1).view(-1, 1)
        confusion_matrix_list = confusion_matrix.tolist()
        formatted_confusion_matrix = [
            [f"{int(item*100)}%" for item in row] for row in confusion_matrix_list
        ]
        headers = [f"Pred_{i}" for i in range(n_classes)]
        print("Confusion matrix:")
        print(
            tabulate(
                formatted_confusion_matrix,
                headers=headers,
                showindex="always",
                tablefmt="pretty",
            )
        )
