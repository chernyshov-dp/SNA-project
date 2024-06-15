import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)[:, None]  # Ensure y has shape (len(data)-seq_length-1, 1)


def calculate_metrics(model, data_loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze(dim=1).to(device)
            output = model(inputs)
            
            # Flatten predictions for each batch
            predictions.extend(output.cpu().numpy().flatten())
            
            # Flatten and concatenate targets to match predictions
            targets.extend(labels.cpu().numpy().reshape(-1))

    targets = np.array(targets)
    predictions = np.array(predictions)

    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    mae_std = np.std(np.abs(predictions - targets))
    mape_std = np.std(np.abs((predictions - targets) / targets))
    r2_std = np.std(r2_score(targets, predictions))

    print(f"{'MAE:':<6} {mae:.4f} ± {mae_std:.2f}")
    print(f"{'MAPE:':<6} {mape:.4f} ± {mape_std:.2f}")
    print(f"{'R²:':<6} {r2:.4f} ± {r2_std:.2f}")

    return mae, r2


def plot_actual_predicted(targets, predictions, save=False, filename="prediction.png"):
    if save:
        plt.figure(figsize=(12, 6))
        plt.plot(targets, label="Ground truth", color="black")
        plt.plot(predictions, label="Predicted", color="blue")
        
        std_dev = np.std(predictions)
        upper_bound = predictions + std_dev
        lower_bound = predictions - std_dev
        
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()
        upper_bound = np.array(upper_bound).flatten()
        lower_bound = np.array(lower_bound).flatten()
        
        plt.fill_between(range(len(predictions)), lower_bound, upper_bound, color="#a3f7bf", alpha=0.2, label="Prediction range")
        plt.title("Ground truth vs Predicted S&P 100")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend(loc="lower right")
        plt.savefig(filename)
        print(f" - Successfully saves as {filename}")


def main():
    SEQ_LENGTH = 90

    INPUT_SIZE = 55
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2

    NUM_EPOCHS = 100
    BATCH_SIZE = 64

    with open("scaled_features.pkl", "rb") as f:
        scaled_features = pickle.load(f)

    # Data
    X, y = create_sequences(scaled_features, SEQ_LENGTH)

    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()

    train_size = int(len(X) * 0.8)

    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE , NUM_LAYERS)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.6f}")
        
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        calculate_metrics(model, test_loader, device)

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            predictions.extend(output.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    plot_actual_predicted(targets, predictions, save=True, filename="prediction.png")


if __name__ == "__main__":
    main()
