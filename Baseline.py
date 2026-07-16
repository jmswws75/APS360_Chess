import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class ChessDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        
        self.X = torch.tensor(data['X'], dtype=torch.int8) 
        self.y = torch.tensor(data['y'], dtype=torch.long)
        print(f"Loadded from {path}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index].float(), self.y[index]
        
class BaselineModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super(BaselineModel, self).__init__()
        input_size = 14*8*8
        output_size = 4096
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
def evaluate(net, loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err_top1 = 0.0
    total_err_top3 = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        
        corr_top1 = torch.argmax(outputs, dim=1) != labels 
        total_err_top1 += int(corr_top1.sum())
        
        _, top3_preds = torch.topk(outputs, 3, dim=1)
        correct_top3 = top3_preds.eq(labels.view(-1, 1).expand_as(top3_preds))
        err_top3 = ~correct_top3.any(dim=1)
        total_err_top3 += int(err_top3.sum())
        
        total_loss += loss.item()
        total_epoch += len(labels)
        
    err_top1 = float(total_err_top1) / total_epoch
    err_top3 = float(total_err_top3) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err_top1, err_top3, loss

def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30, train_loader=None, val_loader=None):
    
    torch.manual_seed(1000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    train_err_top1 = np.zeros(num_epochs)
    train_err_top3 = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err_top1 = np.zeros(num_epochs)
    val_err_top3 = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err_top1 = 0.0
        total_train_err_top3 = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            # Note we cannot use the float cast anymore because we are dealing with
            # Multiple class classification now, and the output will not just be
            # a single number.
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            
            corr_top1 = torch.argmax(outputs, dim=1) != labels
            total_train_err_top1 += int(corr_top1.sum())
            
            _, top3_preds = torch.topk(outputs, 3, dim=1)
            correct_top3 = top3_preds.eq(labels.view(-1, 1).expand_as(top3_preds)) # Check if the label is inside the top 3 predictions
            err_top3 = ~correct_top3.any(dim=1)
            total_train_err_top3 += int(err_top3.sum())
            
            total_train_loss += loss.item()
            total_epoch += len(labels)
            
        train_err_top1[epoch] = float(total_train_err_top1) / total_epoch
        train_err_top3[epoch] = float(total_train_err_top3) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err_top1[epoch], val_err_top3[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        
        print(("Epoch {}: Train Top-1 err: {}, Train Top-3 err: {}, Train loss: {} |"+
               "Val Top-1 Err: {:.4f} | Val Top-3 Err: {:.4f}, Validation loss: {}").format(
                   epoch + 1,
                   train_err_top1[epoch],
                   train_err_top3[epoch],
                   train_loss[epoch],
                   val_err_top1[epoch],
                   val_err_top3[epoch],
                   val_loss[epoch]))
                   
        torch.save(net.state_dict(), f"model_epoch_{epoch+1}.pth")
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,21), train_loss, label="Training Loss", color='blue', linewidth=2)
    plt.plot(range(1,21), val_loss, label="Validation Loss", color='orange', linewidth=2)

    plt.title("Model Loss Over Epochs (4M Positions)")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    # err/acc
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,21), train_err_top1, label="Train Top-1 Error", color='blue', linestyle='-')
    plt.plot(range(1,21), val_err_top1, label="Val Top-1 Error", color='orange', linestyle='-')

    plt.plot(range(1,21), train_err_top3, label="Train Top-3 Error", color='blue', linestyle='-.')
    plt.plot(range(1,21), val_err_top3, label="Val Top-3 Error", color='orange', linestyle='-.')

    plt.title("Model Error Over Epochs (4M Positions)")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig("error_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Curves saved as 'loss_curve.png' and 'error_curve.png'!")

if __name__ == "__main__":
    dataset = ChessDataset("chess_dataset.npz")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    model = BaselineModel(hidden_size=1024)
    train_net(model, batch_size=64, learning_rate=0.01, num_epochs=20, train_loader=train_loader, val_loader=val_loader)