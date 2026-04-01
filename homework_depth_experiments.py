import torch
import matplotlib.pyplot as plt
from fully_connected_basics.datasets import get_mnist_loaders, get_cifar_loaders
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import plot_training_history, count_parameters

device = torch.device("cuda")

train_loader, test_loader = get_mnist_loaders()
# train_loader, test_loader = get_cifar_loaders()

# слои для 1.1
# depth_configs = {
#     "1 слой": [{"type": "linear", "size": 128}],
#     "2 слоя": [
#             {"type": "linear", "size": 256},
#             {"type": "linear", "size": 128}
#         ],
#     "3 слоя": [
#             {"type": "linear", "size": 512},
#             {"type": "linear", "size": 256},
#             {"type": "linear", "size": 128}
#         ],
#     "5 слоёв": [
#             {"type": "linear", "size": 1024},
#             {"type": "linear", "size": 512},
#             {"type": "linear", "size": 256},
#             {"type": "linear", "size": 128},
#             {"type": "linear", "size": 64}
#         ],
#     "7 слоёв": [
#             {"type": "linear", "size": 1024},
#             {"type": "linear", "size": 512},
#             {"type": "linear", "size": 256},
#             {"type": "linear", "size": 128},
#             {"type": "linear", "size": 64},
#             {"type": "linear", "size": 32},
#             {"type": "linear", "size": 16}
#         ]
# }

# слои для 1.2
depth_configs = {
    "5 слоёв": [
            {"type": "linear", "size": 1024},
            {"type": "linear", "size": 512},
            {"type": "linear", "size": 256},
            {"type": "linear", "size": 128},
            {"type": "linear", "size": 64}
        ],

    "5 слоёв (+ dropout)": [
            {"type": "linear", "size": 1024},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 512},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 256},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 128},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 64}
        ],

    "5 слоёв (+ BatchNorm)": [
            {"type": "linear", "size": 1024},
            {"type": "batch_norm"},
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "linear", "size": 128},
            {"type": "batch_norm"},
            {"type": "linear", "size": 64},
            {"type": "batch_norm"}
        ]
}

all_historyes = []

for name, layers in depth_configs.items():
    print(name)
    model = FullyConnectedModel(
        input_size=784,
        # input_size=32*32*3,
        num_classes=10,
        layers = layers).to(device)
    print(f"Параметры {count_parameters(model)}")

    history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=5,
        lr=0.001,
        device="cuda")
    
    all_historyes.append([history, name])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for history, label in all_historyes:
    ax1.plot(history["train_losses"], linestyle="-", label=f"{label} train")
    ax1.plot(history["test_losses"], linestyle="--", label=f"{label} test")

    ax2.plot(history["train_accs"], linestyle="-", label=f"{label} train")
    ax2.plot(history["test_accs"], linestyle="--", label=f"{label} test")

ax1.set_title('Loss')
ax1.legend()

ax2.set_title('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
