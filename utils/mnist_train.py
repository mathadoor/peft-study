from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import mlflow
from mlflow.models import infer_signature
from tqdm import tqdm


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.accumulator = 0
        self.n = 0

    def update(self, value, nums):
        self.accumulator += value
        self.n += nums

    def average(self):
        ret = self.accumulator / self.n
        self.reset()
        return ret

    def reset(self):
        self.accumulator = 0
        self.n = 0


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(64 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)


def initialize_mlflow():
    uri = mlflow.get_tracking_uri()
    if uri is None:
        mlflow.set_tracking_uri('http://127.0.0.1:8080')


def download_mnist():
    return MNIST(root='data', train=True, download=True, transform=ToTensor())


def train_mnist(model, train_loader, optimizer, criterion, acc_meter, loss_meter, epoch):
    acc_meter.reset()
    loss_meter.reset()
    for i, (x, y) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        # Forward Pass
        logit = model(x)
        loss = criterion(logit, y)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Update Metrics
        y_pred = logit.argmax(dim=1)
        acc = (y_pred == y).int().sum()
        acc_meter.update(acc.item(), train_loader.batch_size)
        loss_meter.update(loss.item(), train_loader.batch_size)

    mlflow.log_metric("train_loss", loss_meter.average(), step=epoch)
    mlflow.log_metric("train_accuracy", acc_meter.average(), step=epoch)


def val_mnist(model, val_loader, criterion, acc_meter, loss_meter, epoch):
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(val_loader)):
            logits = model(x)
            loss = criterion(logits, y)
            y_pred = logits.argmax(dim=1)
            acc = (y_pred == y).int().sum()
            acc_meter.update(acc.item(), val_loader.batch_size)
            loss_meter.update(loss.item(), val_loader.batch_size)

        mlflow.log_metric("val_loss", loss_meter.average(), step=epoch)
        mlflow.log_metric("val_accuracy", acc_meter.average(), step=epoch)


def train(optimizer, model, criterion, train_loader, val_loader, epochs):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for epoch in range(epochs):
        train_mnist(model, train_loader, optimizer, criterion, acc_meter, loss_meter, epoch)
        val_mnist(model, val_loader, criterion, acc_meter, loss_meter, epoch)


if __name__ == "__main__":
    # set parameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 0.0001

    # get data
    mnist = download_mnist()
    train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mnist, batch_size=batch_size, shuffle=False)

    # get model, optimizer, and loss function
    model = SimpleCNN()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # initialize mlflow experiment
    initialize_mlflow()
    experiment_string = f"mnist"
    experiment = mlflow.get_experiment_by_name(experiment_string)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_string)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_string)

    # train model
    run_name_string = f"mnist_num_epochs_{num_epochs}_batch_size_{batch_size}_lr_{learning_rate}_wd_{weight_decay}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_string):
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        })
        train(optimizer, model, criterion, train_loader, val_loader, num_epochs)
        print("Logging model")
        for x, y in val_loader:
            with torch.no_grad():
                logit = model(x)
                y_pred = logit.argmax(dim=1)
                signature = infer_signature(x.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
                mlflow.pytorch.log_model(pytorch_model=model,
                                         artifact_path="mnist_model",
                                         signature=signature,
                                         registered_model_name=f"model_{experiment_string}")
            break


