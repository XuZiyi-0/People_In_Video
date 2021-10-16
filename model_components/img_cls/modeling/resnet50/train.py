import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import net
import dataloader
import os

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_sum=0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to('cuda:0').half()
        y = y.to('cuda:0')

        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to("cuda:0").half()
            y = y.to('cuda:0')
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct),test_loss


if __name__ == "__main__":
    model = net.ResNet50().to('cuda:0').half()
    model = torch.nn.DataParallel(model, device_ids=(0,1,2,3))
    learning_rate = 1e-3
    batch_size = 32
    epochs = 50
    log_dir="./weigh"

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, eps=1e-3)
    for epoch in range(50):
        train_loop(dataloader.train_loader,model,loss_fn,optimizer)
        acc,loss = test_loop(dataloader.test_loader,model,loss_fn)
        torch.save(obj=model.state_dict(), f='./weigh/' + 'ep'+str(epoch)+'_acc:'+str(acc)+'_loss:'+str(loss)+'.pkl')
