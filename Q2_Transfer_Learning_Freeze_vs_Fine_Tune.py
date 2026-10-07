#!/usr/bin/env python3
"""Compare frozen-feature training with fine-tuning a pretrained ResNet-18."""

# This script shows a basic transfer-learning example for image classification.
# A beginner-friendly idea: start with a model pre-trained on a large dataset,
# then adapt it to a new task (CIFAR-10) using either freezing or fine-tuning.
# The goal is to compare two ways of reusing the pre-trained model.
#
# How to run this script:
#   1) Open a terminal in this project folder.
#   2) Activate the environment: source .venv-q2/bin/activate
#   3) Run a quick test: ./run_q2_transfer_learning.sh --epochs 1 --train-limit 200 --test-limit 100 --batch-size 32
#   4) Or run the full version: ./run_q2_transfer_learning.sh --epochs 2 --train-limit 1000 --test-limit 500 --batch-size 32
#
# What is happenning: the model learns from a pre-trained backbone and then trains a
# new classifier for CIFAR-10. In the frozen case, the base is not updated. In the
# fine-tuned case, the last block is unfrozen and learns with the classifier.

import argparse
import os
import time

# Put the downloaded PyTorch model cache inside this project folder so the model and
# weights are reused instead of downloading again on every run.
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(__file__), ".torch_cache"))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def choose_device() -> torch.device:
    # Use GPU if available; otherwise use Apple Metal (MPS) on Mac; else CPU.
    # This makes trainning faster when the computer supports a specialized accelerator.
    # In other words, the model should run on the best hardware the laptop has.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(train_limit: int, test_limit: int, batch_size: int, seed: int):
    # This transform prepares each image for ResNet-18.
    # We resize the image to 224x224, convert to tensor, and normalize using the
    # mean and standard deviation used in ImageNet pretraining.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # Keep the dataset in the project folder so we do not download it again and again.
    # If the extracted CIFAR-10 files are already ther, we can reuse them.
    # This saves time and avoids wasting bandwith on repeated downloads.
    data_root = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_root, exist_ok=True)
    cifar_dir = os.path.join(data_root, "cifar-10-batches-py")
    should_download = not os.path.exists(cifar_dir)

    if should_download:
        print("Downloading CIFAR-10 dataset to local data folder ...")
    else:
        print("Using cached CIFAR-10 dataset in local data folder.")

    # Train data: use the training split; test data: use the test split.
    # We only take a subset to speed up experments and keep the code easy to read.
    train_data = datasets.CIFAR10(root=data_root, train=True, download=should_download, transform=transform)
    test_data = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    if train_limit <= 0 or test_limit <= 0:
        raise ValueError("Training and test sample limits must be positive.")

    # Random subsets help keep the run small and manageable for homework and debugging.
    # We use a fixed seed so the same samples are selected for fair comparision.
    # This is important because we want both methods to see the same data setup.
    train_indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed))[:train_limit]
    test_indices = torch.randperm(len(test_data), generator=torch.Generator().manual_seed(seed + 1))[:test_limit]
    train_subset = Subset(train_data, train_indices.tolist())
    test_subset = Subset(test_data, test_indices.tolist())

    # DataLoader groups images into batches. The batches are shuffled during training.
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def make_model(experiment: str, device: torch.device) -> nn.Module:
    # These pretraind weights already recognize common visual patterns, even if the spelling is a bit off here.
    # ResNet-18 is a strong backbone for image tasks and is already trained on ImageNet.
    # The model knows basic things like edges, colors, and shapes before this homework starts.
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    # In the frozen case, all base layers are locked and only the final classifier is trained.
    # This saves time and uses the learned low/mid-level visual features.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Change the last layer from 1000 ImageNet classes to 10 CIFAR-10 classes.
    model.fc = nn.Linear(model.fc.in_features, 10)
    if experiment == "fine-tuned":
        # For fine-tuning, we unfreeze the final convolutional block.
        # This allows the top of the network to adapt to the new dataset.
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    return model.to(device)


def run_experiment(
    experiment: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    seed: int,
) -> dict:
    # Fix the random seed so the train/test subset order is consistent and the experiment
    # is easier to compare from one run to another.
    # This helps make the results more reproducible, even though many runs still vary a little.
    torch.manual_seed(seed)
    # Reuse the same shuffled sample order to make the experment comparison fair.
    # The goal is to keep the two methods playing on the same stage.
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)
    model = make_model(experiment, device)
    # Count only weights that the optimizer can update in this experiment.
    # This tells us how much of the model actualy learns on our task.
    # Fewer trainable params means the model is more constrained.
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
    )
    loss_function = nn.CrossEntropyLoss()
    epoch_losses = []

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        # In the frozen setup, the base model is kept in eval mode so its batch stats stay fixed.
        # In the fine-tuned setup, only the deeper blocks are kept in eval mode to preserve
        # pretrained statistics while allowing the last block to adapt.
        if experiment == "frozen":
            model.eval()
            model.fc.train()
        else:
            for frozen_block in (model.bn1, model.layer1, model.layer2, model.layer3):
                frozen_block.eval()
        loss_total = 0.0
        sample_total = 0
        for images, labels in train_loader:
            # Move images and labels to the active device, such as GPU or MPS.
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.size(0)
            sample_total += labels.size(0)
        epoch_losses.append(loss_total / sample_total)
        print(f"{experiment}: epoch {epoch + 1}/{epochs}, loss={epoch_losses[-1]:.4f}")
    training_seconds = time.perf_counter() - start_time

    # Evaluate the model on the test set after training.
    model.eval()
    correct = 0
    sample_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            sample_total += labels.size(0)

    return {
        "method": experiment,
        "trainable_parameters": trainable_parameters,
        "training_seconds": training_seconds,
        "accuracy": correct / sample_total * 100,
        "losses": epoch_losses,
    }


def main() -> None:
    # This is the main entry point of the program.
    # It lets a user set how many epochs to train, how many images to use,
    # and how big each mini-batch should be. This is useful for a beginner because
    # smaller values make debugging faster and easier to understand.
    parser = argparse.ArgumentParser(description="Compare frozen ResNet-18 features with fine-tuning.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per experiment (default: 2).")
    parser.add_argument("--train-limit", type=int, default=1000, help="Training images to use (default: 1000).")
    parser.add_argument("--test-limit", type=int, default=500, help="Test images to use (default: 500).")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per batch (default: 32).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    arguments = parser.parse_args()

    # We check the values her to avoid bad configs. If the user gives 0 or a negitive number,
    # the program stops early and tells them why.
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")

    # Pick the device for training. On a Mac, this may use MPS; on a PC with NVIDIA GPU, it may use CUDA.
    # If no GPU is availabe, the code falls back to CPU.
    device = choose_device()
    print(f"Device: {device}")
    print("Downloading CIFAR-10 and pretrained ResNet-18 weights on the first run if needed.")

    # This function builds the training and test loaders using the selected subset sizes.
    # It also keeps the dataset in a local folder, so the data does not need to be downloaded again.
    train_loader, test_loader = build_loaders(
        arguments.train_limit,
        arguments.test_limit,
        arguments.batch_size,
        arguments.seed,
    )

    # Run both experiments using the exact same data subset and same epoch count.
    # This makes the comparision fair because both methods are tested under similiar condtions.
    results = [
        run_experiment("frozen", train_loader, test_loader, arguments.epochs, device, arguments.seed),
        run_experiment("fine-tuned", train_loader, test_loader, arguments.epochs, device, arguments.seed),
    ]

    # Print the final result table. This is the main answer for the homework question.
    # It show how many params each model can learn, how long the training took,
    # and how accurate the model was on the test set.
    print("\nComparison (same dataset subset and number of epochs):")
    print("| Method | Trainable parameters | Training time (s) | Test accuracy |")
    print("|---|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['method']} | {result['trainable_parameters']:,} | "
            f"{result['training_seconds']:.2f} | {result['accuracy']:.2f}% |"
        )

    # Plot the loss curve for both methods. This helps a beginner see how the loss changes
    # over training and understand which method learns faster or more stably.
    for result in results:
        plt.plot(range(1, arguments.epochs + 1), result["losses"], marker="o", label=result["method"])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Frozen Features vs. Fine-Tuning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = "q2_training_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training-loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()