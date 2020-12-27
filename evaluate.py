import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from recognizer.model import KanjiRecognizer


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def run(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH = './cifar_net.pth'
    model = KanjiRecognizer(input_dimensions=32, output_dimensions=10)
    model.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # print images
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images with kanji on them.')
    parser.add_argument('-c', '--character-count', type=int, default=0,
                        help='amount of characters to generate')
    parser.add_argument('-s', '--repetition-count', type=int, default=10,
                        help='amount of sets to generate for each character')
    parser.add_argument('-o', '--output-path', type=str, default='generated',
                        help='path to the folder where generated images are going to be saved')
    parser.add_argument('-f', '--font', type=str, default='/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc',
                        help='path to font to use')
    parser.add_argument('-b', '--background-image-folder', type=str,
                        help='path to a folder containing background images')
    args = parser.parse_args()

    run(args)
