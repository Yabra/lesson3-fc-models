import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from fully_connected_basics.trainer import run_epoch
from fully_connected_basics.datasets import get_mnist_loaders, get_cifar_loaders
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.utils import plot_training_history, count_parameters

def set_dropout(model, rate):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = rate

def set_batchnorm_momentum(model, momentum):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = momentum

# слегка изменим train_model, чтобы оптимизатору можно было задать параметр weight_decay
# а также можно было применить адаптивные техники Dropout и BatchNorm
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', weight_decay=None, dynamic_dropout=False, dynamic_batchnorm=False):
    criterion = nn.CrossEntropyLoss()

    if weight_decay is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        # добавляем изменения Dropout при смене эпох
        if dynamic_dropout:
            current_dropout = 0.5 * (1 - epoch / epochs)
            set_dropout(model, current_dropout)

        # добавляем изменения BatchNorm при смене эпох
        if dynamic_batchnorm:
            current_momentum = 0.1 * 0.5 * (epoch / epochs)
            set_batchnorm_momentum(model, current_momentum)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    } 


device = torch.device("cuda")

train_loader, test_loader = get_mnist_loaders()


# слои для 3.1
# depth_configs = {
#     "L2": [
#         {"type": "linear", "size": 256},
#         {"type": "linear", "size": 128},
#         {"type": "linear", "size": 64}
#     ],
    
#     "Без регуляризации": [
#             {"type": "linear", "size": 256},
#             {"type": "linear", "size": 128},
#             {"type": "linear", "size": 64}
#         ],

#     "Dropout 0.1": [
#             {"type": "linear", "size": 256},
#             {"type": "dropout", "rate": 0.1},
#             {"type": "linear", "size": 128},
#             {"type": "dropout", "rate": 0.1},
#             {"type": "linear", "size": 64}
#         ],

#     "Dropout 0.3": [
#             {"type": "linear", "size": 256},
#             {"type": "dropout", "rate": 0.3},
#             {"type": "linear", "size": 128},
#             {"type": "dropout", "rate": 0.3},
#             {"type": "linear", "size": 64}
#         ],

#     "Dropout 0.5": [
#             {"type": "linear", "size": 256},
#             {"type": "dropout", "rate": 0.5},
#             {"type": "linear", "size": 128},
#             {"type": "dropout", "rate": 0.5},
#             {"type": "linear", "size": 64}
#         ],

#     "BatchNorm": [
#             {"type": "linear", "size": 256},
#             {"type": "batch_norm"},
#             {"type": "linear", "size": 128},
#             {"type": "batch_norm"},
#             {"type": "linear", "size": 64}
#         ],

#     "BatchNorm + Dropout": [
#             {"type": "linear", "size": 256},
#             {"type": "batch_norm"},
#             {"type": "dropout", "rate": 0.3},
#             {"type": "linear", "size": 128},
#             {"type": "batch_norm"},
#             {"type": "dropout", "rate": 0.3},
#             {"type": "linear", "size": 64}
#         ]
# }

depth_configs = {
    "Adaptive": [
        {"type": "linear", "size": 256},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},

        {"type": "linear", "size": 128},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},

        {"type": "linear", "size": 64},
        {"type": "batch_norm"},
        {"type": "relu"}
    ],
}

all_historyes = []

for name, layers in depth_configs.items():
    print(name)
    model = FullyConnectedModel(
        input_size=784,
        # input_size=32*32*3,
        num_classes=10,
        layers = layers,
        ).to(device)
    print(f"Параметры {count_parameters(model)}")

    history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=5,
        lr=0.001,
        device="cuda",
        weight_decay=(1e-4 if name == "L2" else None),
        dynamic_dropout=name == "Adaptive",
        dynamic_batchnorm=name == "Adaptive",
        )
    
    all_historyes.append([history, name, model])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for history, label, model in all_historyes:
    weights = []

    ax1.plot(history["train_accs"], linestyle="-", label=f"{label} train")
    ax1.plot(history["test_accs"], linestyle="--", label=f"{label} test")

    for p in model.parameters():
        weights.extend(p.detach().cpu().numpy().flatten())

    ax2.hist(weights, bins=50, alpha=0.5, label=label)
    


ax1.set_title("Accuracy")
ax1.legend()

ax2.set_title("Weight distribution")
ax2.legend()

plt.tight_layout()

plt.show()
