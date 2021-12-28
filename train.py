import numpy as np
import torch
import torch.optim as optim

from helpers import make_batch, adjust_learning_rate
from model import Net
from loss import modulated_loss

def train(model, optimizer, epoch, device, steps, batch_size, criterion):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for _ in range(0, steps):
        images, target = make_batch(batch_size)

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        iou = criterion(output, target)
        
        iou = torch.mean(iou)
        running_iou += iou.item()

        optimizer.zero_grad()
        iou.backward()
        optimizer.step()

        running_loss += iou.item()

    print(epoch)
    print(running_loss / steps)
    print(running_iou / steps)

def main():
    model = Net()

    # Part I - Train model to localize spaceship on images containing spaceship
    print("Start localization training")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), eps=1e-07)

    criterion = modulated_loss

    epochs = 10
    steps_per_epoch = 2000
    batch_size = 64

    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch)
        train(model, optimizer, epoch, device, steps_per_epoch, batch_size, criterion)

    path = F'model.pth.tar'
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
