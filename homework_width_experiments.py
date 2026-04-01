import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fully_connected_basics.datasets import get_mnist_loaders, get_cifar_loaders
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import plot_training_history, count_parameters

device = torch.device("cuda")

train_loader, test_loader = get_mnist_loaders()

# код для 2.1

# width_configs = {
#     "Узкие": [
#             {"type": "linear", "size": 64},
#             {"type": "linear", "size": 32},
#             {"type": "linear", "size": 16}
#         ],
#     "Средние": [
#             {"type": "linear", "size": 256},
#             {"type": "linear", "size": 128},
#             {"type": "linear", "size": 64}
#         ],
#     "Широкие": [
#             {"type": "linear", "size": 1024},
#             {"type": "linear", "size": 512},
#             {"type": "linear", "size": 256}
#         ],
#     "Очень широкие": [
#             {"type": "linear", "size": 2048},
#             {"type": "linear", "size": 1024},
#             {"type": "linear", "size": 512}
#         ]
# }

# all_historyes = []
# all_time_results = []

# for name, layers in width_configs.items():
#     print(name)
#     model = FullyConnectedModel(
#         input_size=784,
#         # input_size=32*32*3,
#         num_classes=10,
#         layers = layers).to(device)
#     print(f"Параметры {count_parameters(model)}")

#     start = time.time()
#     history = train_model(
#         model,
#         train_loader,
#         test_loader,
#         epochs=5,
#         lr=0.001,
#         device="cuda")
    
#     all_time_results.append(f"Модель {name} обучилась за {time.time() - start} секунд")
#     all_historyes.append([history, name])


# print("\n".join(all_time_results))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# for history, label in all_historyes:
#     ax1.plot(history["train_losses"], linestyle="-", label=f"{label} train")
#     ax1.plot(history["test_losses"], linestyle="--", label=f"{label} test")

#     ax2.plot(history["train_accs"], linestyle="-", label=f"{label} train")
#     ax2.plot(history["test_accs"], linestyle="--", label=f"{label} test")

# ax1.set_title('Loss')
# ax1.legend()

# ax2.set_title('Accuracy')
# ax2.legend()

# plt.tight_layout()
# plt.show()



# Код для 2.2
width_configs = {
    "Расширенные": [
            {"type": "linear", "size": 64},
            {"type": "linear", "size": 128},
            {"type": "linear", "size": 256}
        ],
    "Суженные": [
            {"type": "linear", "size": 256},
            {"type": "linear", "size": 128},
            {"type": "linear", "size": 64}
        ],
    "Постоянные": [
            {"type": "linear", "size": 128},
            {"type": "linear", "size": 128},
            {"type": "linear", "size": 512}
        ]
}

lrs = [0.0001, 0.001, 0.01]
results = np.zeros((len(width_configs), len(lrs)))

for i, (name, layers) in enumerate(width_configs.items()):
    for j, lr in enumerate(lrs):
        print(name, lr)
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers = layers).to(device)

        history = train_model(
            model,
            train_loader,
            test_loader,
            epochs=5,
            lr=lr,
            device="cuda")
        
        results[i, j] = history["test_accs"][-1]

plt.figure(figsize=(8, 6))

sns.heatmap(results, annot=True, fmt=".3f",
            xticklabels=lrs,
            yticklabels=width_configs.keys())

plt.xlabel("Lerning rate")
plt.ylabel("Слои")
plt.show()
