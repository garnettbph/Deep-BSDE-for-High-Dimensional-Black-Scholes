import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

from environment import HighDimBlackScholes
from model import DeepBSDE

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 3000
    LEARNING_RATE = 1e-2

    market = HighDimBlackScholes(dim=100, device=device)
    model = DeepBSDE(market).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000], gamma=0.1)

    true_price = market.analytical_pricing()
    print(f"Analytical True Price (Benchmark): {true_price:.4f}\n")

    loss_history = []
    y0_history = []

    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        X, dW = market.generate_paths(BATCH_SIZE)
        
        Y_terminal_pred = model(X, dW)
        
        Y_terminal_true = market.terminal_condition(X[:, -1, :]).unsqueeze(1)
        
        loss = nn.MSELoss()(Y_terminal_pred, Y_terminal_true)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        y0_history.append(model.Y_0.item())
        
        if (epoch + 1) % 500 == 0:
            error = abs(model.Y_0.item() - true_price)
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | Loss: {loss.item():.4f} | "
                  f"Predicted Y_0: {model.Y_0.item():.4f} | Error to Linear Benchmark: {error:.4f}")

    print(f"Final Predicted Price: {model.Y_0.item():.4f}")
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].plot(loss_history, color='blue', alpha=0.7)
    ax[0].set_yscale('log')
    ax[0].set_title('Training Loss (MSE)')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss (Log Scale)')
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(y0_history, color='green', alpha=0.8, label='Deep BSDE Prediction')
    ax[1].axhline(y=true_price, color='red', linestyle='--', label='Analytical Benchmark')
    ax[1].set_title('Option Price Convergence ($Y_0$)')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Price')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    learned_Z0 = model.Z_0.detach().cpu().numpy().flatten()
    ax[2].bar(range(market.d), learned_Z0, color='purple', alpha=0.7)
    ax[2].set_title('Learned Initial Deltas ($Z_0$) per Asset')
    ax[2].set_xlabel('Asset Dimension (1 to 100)')
    ax[2].set_ylabel('Delta ($Z_0$)')
    ax[2].grid(True, alpha=0.3)
    ax[2].set_ylim([0, max(learned_Z0) * 1.5]) 

    plt.tight_layout()
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
