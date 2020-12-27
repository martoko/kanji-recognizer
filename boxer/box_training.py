import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from matplotlib import pyplot
from torch import Tensor

import datasets
from boxer.box_model import KanjiBoxer


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability() != (3, 0) and
                                    torch.cuda.get_device_capability()[0] >= 3 else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.Kanji(args.fonts_folder, args.background_images_folder, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128)
    testset = datasets.Kanji(args.fonts_folder, args.background_images_folder, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    PATH = './box_saved_model.pt'
    model = KanjiBoxer(input_dimensions=32).to(device)
    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_length = 10
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; datasets is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.float().to(device) / 32

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % epoch_length == (epoch_length - 1):  # print every 20 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / epoch_length:.5f}')
            running_loss = 0.0

        if i >= 1000:
            break

    print('Finished Training')

    torch.save(model.state_dict(), PATH)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.data
            total += labels.size(0)
            correct += ((predicted * 32).int() == labels.int()).sum().item()

            if total >= 10000:
                break

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels: Tensor = labels.to(device)

    outputs = model(images)

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # imshow(torchvision.utils.make_grid(images[:10]))
    # grid = torchvision.utils.make_grid((images[0] * (0.5, 0.5, 0.5) + (0.5, 0.5, 0.5)))
    # # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig, axes = pyplot.subplots(nrows=2, ncols=5)

    for i in range(5):
        im = Image.fromarray(np.transpose(((images[i] / 2 + 0.5).numpy() * 255).astype(np.uint8), (1, 2, 0)))
        drawing = ImageDraw.Draw(im)
        drawing.rectangle(list(labels[i].cpu().detach().numpy()), outline='red')
        axes[0, i].axis('off')
        axes[0, i].imshow(im)

        im = Image.fromarray(np.transpose(((images[i] / 2 + 0.5).numpy() * 255).astype(np.uint8), (1, 2, 0)))
        drawing = ImageDraw.Draw(im)
        drawing.rectangle(list((outputs[i] * 32).cpu().detach().numpy()), outline='red')
        axes[1, i].axis('off')
        axes[1, i].imshow(im)
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images with kanji on them.')
    parser.add_argument('-c', '--character-count', type=int, default=0,
                        help='amount of characters to generate')
    parser.add_argument('-s', '--repetition-count', type=int, default=10,
                        help='amount of sets to generate for each character')
    parser.add_argument('-o', '--output-path', type=str, default='generated',
                        help='path to the folder where generated images are going to be saved')
    parser.add_argument('-f', '--fonts-folder', type=str, default='fonts',
                        help='path to a folder containing fonts')
    parser.add_argument('-b', '--background-images-folder', type=str,
                        help='path to a folder containing background images')
    args = parser.parse_args()

    run(args)
