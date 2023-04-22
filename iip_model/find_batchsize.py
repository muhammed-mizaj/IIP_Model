import torch
import typing as t
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights


from time import sleep

# dataset information
IMAGE_SHAPE = (3, 416, 416)
NUM_CLASSES = 5
DATASET_SIZE = 1440


def get_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: t.Tuple[int, int, int],
    output_shape: t.Tuple[int],
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    print("Test batch size")
    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(
                    *(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
            print(f"\tTesting batch size {batch_size}")
            sleep(3)
        except RuntimeError:
            print(f"\tOOM at batch size {batch_size}")
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    print(f"Final batch size {batch_size}")
    return batch_size


def get_datasets(batch_size: int, num_workers: int = 2):
    train_ds = DataLoader(
        datasets.FakeData(
            size=DATASET_SIZE,
            image_size=IMAGE_SHAPE,
            num_classes=NUM_CLASSES,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_ds = DataLoader(
        datasets.FakeData(
            size=200,
            image_size=IMAGE_SHAPE,
            num_classes=NUM_CLASSES,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_ds, test_ds


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.output_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_features=1000, out_features=NUM_CLASSES),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor):
        outputs = self.resnet(inputs)
        outputs = self.output_layer(outputs)
        return outputs


def train(
    model: nn.Module,
    optimizer: torch.optim,
    train_ds: DataLoader,
    device: torch.device,
):
    model.train()
    train_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(tqdm(train_ds, desc="Train")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    return {
        "loss": train_loss / len(train_ds),
        "accuracy": 100.0 * correct / len(train_ds.dataset),
    }


def test(model: nn.Module, test_ds: DataLoader, device: torch.device):
    with torch.no_grad():
        model.eval()
        test_loss, correct = 0, 0
        for data, target in tqdm(test_ds, desc="Test"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    return {
        "loss": test_loss / len(test_ds),
        "accuracy": 100.0 * correct / len(test_ds.dataset),
    }


def main(epochs: int = 2):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    batch_size = get_batch_size(
        model=ResNet(),
        device=device,
        input_shape=IMAGE_SHAPE,
        output_shape=(NUM_CLASSES,),
        dataset_size=DATASET_SIZE,
    )

    train_ds, test_ds = get_datasets(batch_size=batch_size)
    model = ResNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_result = train(
            model=model, optimizer=optimizer, train_ds=train_ds, device=device
        )
        test_result = test(model=model, test_ds=test_ds, device=device)
        print(
            f'Train loss: {train_result["loss"]:.04f}\t'
            f'accuracy: {train_result["accuracy"]:.2f}%\n'
            f'Test loss: {test_result["loss"]:.04f}\t'
            f'accuracy: {test_result["accuracy"]}%'
        )


if __name__ == "__main__":
    main()
