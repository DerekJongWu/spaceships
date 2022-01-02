import numpy as np
import torch
import torch.optim as optim

from helpers import make_batch
from model import Net
from loss import modulated_loss

def train(model, optimizer, epoch, device, steps, batch_size, criterion, classify):
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_iou_loss = 0.0

    for _ in range(0, steps):
        images, target = make_batch(batch_size)

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        loss, l_ship, l_bbox = criterion(output, target)
        loss, l_ship, l_bbox = torch.mean(loss), torch.mean(l_ship), torch.mean(l_bbox)
        if classify: 
            l_ship.backward()
        else: 
            l_bbox.backward() 
        optimizer.step()

        running_loss += loss.item()
        running_class_loss += torch.mean(l_ship)
        running_iou_loss += torch.mean(l_bbox)

    print(epoch)
    print(f'Loss: {running_loss / steps}')
    print(f'Class Loss: {running_class_loss / steps}')
    print(f'IOU loss: {running_iou_loss/steps}')

def main():
    model = Net()

    # Part I - Train model to localize spaceship on images containing spaceship
    print("Start classification training")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = 0.001)

    criterion = modulated_loss

    epochs = 5
    steps_per_epoch = 500
    batch_size = 64

    for epoch in range(0, epochs):
        train(model, optimizer, epoch, device, steps_per_epoch, batch_size, criterion, classify = True)


    print("Start localization training") 

    for param in model.convnet.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = False

    batch_size = 64
    steps_per_epoch = 3000
    epochs = 40

    optimizer = optim.Adam(model.parameters(), eps=1e-07)

    for epoch in range(0, epochs):
        train(model, optimizer, epoch, device, steps_per_epoch, batch_size, criterion, classify = False)


    path = F'model.pth.tar'
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
