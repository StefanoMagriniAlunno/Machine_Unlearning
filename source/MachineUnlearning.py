import random
from typing import List

import torch
import torch.utils.data
from pandas import DataFrame
from tabulate import tabulate  # type: ignore
from tqdm.notebook import tqdm


def train(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    epochs: int,
    n_layers: int,
    w: float,
    device: torch.device,
    classes: List[int],
):
    """This function trains a model to classify the new classes and to forget the old ones.

    Args:
        model (torch.nn.Module): the model to train
        loader (torch.utils.data.DataLoader): loader of new classes, return target -1 for old classes
        epochs (int): number of epochs
        n_layers (int): number of layers to train in each epoch
        w (float): weight of the imitation loss
    """

    # clear the cache
    torch.cuda.empty_cache()

    # prepare the model
    model.to(device)
    model.train()
    model.zero_grad()

    print(" Num of parameters = ", sum(p.numel() for p in model.parameters()))
    print("    Num of data    = ", len(loader.dataset))
    print("    Batch size     = ", loader.batch_size)
    print("  Num of classes   = ", len(classes))
    print("     Classes       = ", classes)

    n_classes = len(classes)

    # DATAFRAME details:
    # - epoch: the epoch number
    # - accuracy: the general accuracy of the model
    # - confidence: mean variance of log_softmax of the new classes
    # - hiding: the total variance of the softmax of the old classes (1 is maximum)
    df = DataFrame(columns=["epoch", "accuracy", "confidence", "hiding"])
    df["epoch"] = range(epochs)
    df = df.set_index("epoch")

    # save a reference to the trainable layers
    layers = [layer for layer in model.parameters() if layer.requires_grad]

    # define loss for classification
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=-1
    )  # exclude the old class (with label -1)

    # define the confusion matrix
    confusion_matrix = torch.zeros((n_classes, n_classes), device=device)

    # training loop
    progress_epoch = tqdm(range(epochs), desc="Epochs", leave=False)
    for e in progress_epoch:

        # params for the dataframe
        accuracy: float = 0.0
        confidence: float = 0.0
        hiding: float = 0.0
        # params for the progress bar
        loss_epoch: float = 0.0
        class_epoch: float = 0.0
        imitation_epoch: float = 0.0

        # fine-tuning preparation
        epoch_layers_idx = random.sample(range(len(layers)), n_layers)
        for layer in layers:
            layer.requires_grad = False
        for i in epoch_layers_idx:
            layers[i].requires_grad = True
        optimizer = torch.optim.Adam(
            [{"params": layers[i]} for i in epoch_layers_idx],
        )

        x: torch.Tensor
        y: torch.Tensor
        progress_batch = tqdm(loader, desc="Batches", total=len(loader), leave=False)
        for x, y in progress_batch:
            x, y = x.to("cuda"), y.to("cuda")

            # compute the output of the model
            y_: torch.Tensor = model(x)

            # classification loss, ignore the old class
            class_loss: torch.Tensor = loss_fn(y_, y)

            # imitation learning
            generated = torch.nn.functional.softmax(
                y_[y == -1], dim=1
            )  # distribuzione generata
            ground = torch.nn.functional.softmax(
                y_[y != -1], dim=1
            )  # distribuzione ground truth

            if len(generated) > 0:
                # compute first momentum distance
                first_momentum_distance = (
                    torch.norm(generated.mean(dim=0) - ground.mean(dim=0), p="fro") ** 2
                )

                # compute second momentum distance
                second_momentum_distance = torch.norm(
                    torch.cov(generated.T, correction=0)
                    - torch.cov(ground.T, correction=0),
                    p="fro",
                )

                # imitation loss is the square of difference between the two momentum
                imitation_loss = first_momentum_distance + second_momentum_distance
            else:
                imitation_loss = torch.tensor(0.0)

            # total loss as combination of classification and imitation loss
            loss = class_loss + w * imitation_loss

            # compute the loss of the model
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # compute confusion matrix
            for i in range(len(y)):
                if y[i] != -1:
                    confusion_matrix[y[i], y_[i].argmax().item()] += 1

            # compute the local accuracy
            accuracy += (y_[y != -1].argmax(dim=1) == y[y != -1]).sum().item() / len(
                y[y != -1]
            )
            # compute the local confidence
            if torch.any(y != -1):
                confidence += (
                    torch.nn.functional.log_softmax(y_[y != -1], dim=1)
                    .var(dim=1)
                    .mean()
                    .item()
                )
            # compute the local hiding
            if torch.count_nonzero(y == -1) > 1:
                hiding += (
                    torch.nn.functional.softmax(y_[y == -1], dim=1)
                    .var(dim=0)
                    .sum()
                    .item()
                )

            loss_epoch += loss.item()
            class_epoch += class_loss.item()
            imitation_epoch += imitation_loss.item()

        # report data in dataframe
        df.loc[e, "accuracy"] = accuracy / len(loader)
        df.loc[e, "confidence"] = confidence / len(loader)
        df.loc[e, "hiding"] = hiding / len(loader)

        # report data in progress bar
        loss_epoch /= len(loader)
        class_epoch /= len(loader)
        imitation_epoch /= len(loader)
        progress_epoch.set_postfix(
            loss=loss_epoch,
            classification=class_epoch,
            imitation=imitation_epoch,
        )

    # print the confusion matrix
    confusion_matrix /= torch.sum(confusion_matrix, dim=1).view(-1, 1)
    confusion_matrix_list = confusion_matrix.tolist()
    # la confusion matrix mostra le percentuali come numeri interi
    formatted_confusion_matrix = [
        [f"{int(item*100)}%" for item in row] for row in confusion_matrix_list
    ]
    headers = [f"{i}" for i in classes]
    print("Confusion matrix:")
    print(
        tabulate(
            formatted_confusion_matrix,
            headers=headers,
            showindex="always",
            tablefmt="pretty",
        )
    )

    # print the dataframe
    print(tabulate(df, headers="keys", tablefmt="pretty"))


def test(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    classes: List[int],
):
    # clear the cache
    torch.cuda.empty_cache()

    # prepare the model
    model.to(device)
    model.eval()
    model.zero_grad()

    print(" Num of parameters = ", sum(p.numel() for p in model.parameters()))
    print("    Num of data    = ", len(loader.dataset))
    print("    Batch size     = ", loader.batch_size)
    print("  Num of classes   = ", len(classes))
    print("     Classes       = ", classes)

    n_classes = len(classes)

    # creo una matrice con tutti i risultati del modello nella ram
    results: List[torch.Tensor] = [
        torch.empty((0, n_classes), device="cpu") for _ in range(n_classes)
    ]
    fake_results: List[torch.Tensor] = [
        torch.empty((0, n_classes), device="cpu") for _ in range(n_classes)
    ]

    # test loop
    x: torch.Tensor
    y: torch.Tensor
    for x, y in tqdm(loader, desc="Testing", leave=False):
        x, y = x.to(device), y.to(device)

        # compute the output of the model
        y_: torch.Tensor = model(x)

        # save the results
        for i in range(n_classes):
            results[i] = torch.cat([results[i], y_[y == i].detach().cpu()])
            # inserisco i dati in fake_results per assegnazione verosimile
            fake_results[i] = torch.cat(
                [
                    fake_results[i],
                    y_[torch.logical_and(y == -1, y_.argmax(dim=1) == i)]
                    .detach()
                    .cpu(),
                ]
            )

    # compute the accuracy
    accuracy = 0.0
    for i in range(n_classes):
        accuracy += (results[i].argmax(dim=1) == i).sum().item() / len(results[i])
    accuracy /= n_classes
    print("Accuracy: ", accuracy)

    # compute confidence
    confidence = 0.0
    for i in range(n_classes):
        confidence += (
            torch.nn.functional.log_softmax(results[i], dim=1).var(dim=1).mean().item()
        )
    confidence /= n_classes
    print("Confidence: ", confidence)

    # compute the confusion matrix
    results_softmax = [torch.nn.functional.softmax(result, dim=1) for result in results]
    confusion_matrix = torch.zeros((n_classes, n_classes), device=device)
    for i in range(n_classes):
        confusion_matrix[i, :] = torch.mean(results_softmax[i], dim=0)

    # print the confusion matrix
    formatted_confusion_matrix = [
        [f"{int(item*100)}%" for item in row] for row in confusion_matrix
    ]
    headers = [f"{i}" for i in classes]
    print("Confusion matrix:")
    print(
        tabulate(
            formatted_confusion_matrix,
            headers=headers,
            showindex="always",
            tablefmt="pretty",
        )
    )

    # compute hiding
    fake_results_logsoftmax = [
        torch.nn.functional.log_softmax(result, dim=1) for result in fake_results
    ]
    results_logsoftmax = [
        torch.nn.functional.log_softmax(result, dim=1) for result in results
    ]
    fake_covariance = [
        torch.cov(fake_res.T, correction=-1) for fake_res in fake_results_logsoftmax
    ]
    results_covariance = [torch.cov(res.T, correction=-1) for res in results_logsoftmax]
    hiding_cov = [
        torch.norm(fake_cov - res_cov, p="fro").item() / (n_classes * n_classes)
        for fake_cov, res_cov in zip(fake_covariance, results_covariance)
    ]
    # KL con target la distribuzione dei target nel dataset e oggetto la distribuzione dei target di fake
    global_dist = torch.tensor(
        [len(fake_results[i]) for i in range(len(fake_results))], dtype=torch.float32
    )
    target_dist = torch.tensor(
        [len(results[i]) for i in range(len(fake_results))], dtype=torch.float32
    )
    global_dist /= torch.sum(global_dist)
    target_dist /= torch.sum(target_dist)

    # hiding distribution is the kl divergence with target target_dist
    print(
        "Hiding - distribution: ",
        torch.nn.functional.kl_div(
            torch.log(target_dist)[None, :], global_dist[None, :]
        ).item(),
    )
    print("Hiding - covariance: ", [int(hcov * 100) / 100 for hcov in hiding_cov])
