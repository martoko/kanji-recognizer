import argparse
import os
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from datasets import RecognizerDataset
from model import KanjiRecognizer

cuda_is_available = torch.cuda.is_available() and torch.cuda.get_device_capability() != (3, 0) and \
                    torch.cuda.get_device_capability()[0] >= 3


def run(args):
    wandb.init(project="qanji", config=args)
    device = torch.device("cuda" if cuda_is_available else "cpu")

    def denormalize(img):
        return img / 0.5 + 0.5

    def imshow(img):
        npimg = denormalize(img).numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = RecognizerDataset(args.fonts_folder, args.background_images_folder, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)
    testset = RecognizerDataset(args.fonts_folder, args.background_images_folder, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)
    wandb.config.update({"dataset": trainset.id})

    model = KanjiRecognizer(input_dimensions=32, output_dimensions=len(trainset.characters)).to(device)
    wandb.watch(model)
    if args.input_path is not None and os.path.exists(args.input_path):
        print(f"Loading checkpoint from {args.input_path}")
        model.load_state_dict(torch.load(args.input_path))

    def validate():
        with torch.no_grad():
            failure_cases = []
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                for prediction, label, image, output in zip(predictions, labels, images, outputs):
                    if prediction != label:
                        failure_cases += [{
                            "image": image,
                            "prediction": testset.characters[prediction],
                            "label": testset.characters[label],
                            "confidence": output[label]
                        }]

                if total >= len(testset.characters):
                    break
            wandb.log({"failure_cases": [wandb.Image(
                case["image"],
                caption=f"Prediction: {case['prediction']} Truth: {case['label']}"
            ) for case in sorted(failure_cases, key=lambda item: item['confidence'])[:8]]})
            return 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    running_loss = 0.0
    time_of_last_report = time.time()
    log_frequency = 10  # report every X seconds
    batches_since_last_report = 0
    begin_training_time = time.time()
    for current_batch, data in enumerate(trainloader, 1):
        # get the inputs; datasets is a list of [inputs, labels]
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        batches_since_last_report += 1
        if time.time() - time_of_last_report > log_frequency:
            print(f"[{current_batch + 1:5d}] loss: {running_loss / batches_since_last_report:.3f}")
            wandb.log({
                "train/loss": running_loss / batches_since_last_report,
                "train/accuracy": validate(),
                "batch": current_batch,
                "sample": current_batch * args.batch_size
            })
            running_loss = 0.0
            batches_since_last_report = 0
            time_of_last_report = time.time()

            if time.time() - begin_training_time > args.training_time * 60:
                break

    print("Finished Training")

    if args.output_path is not None:
        # TODO: Save a full checkpoint, allow resuming, this includes saving batch/samples processed so far
        print(f"Saving model to {args.output_path}")
        pathlib.Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output_path)

    print("Accuracy of the network: %d %%" % validate())

    def log_examples(count=10):
        with torch.no_grad():
            dataiter = iter(trainloader)
            images, labels = dataiter.next()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            images = images.cpu()
            labels = labels.cpu()
            predictions = predictions.cpu()

            # print images
            examples = min(count, args.batch_size)
            wandb.log({
                "examples": [wandb.Image(
                    image,
                    caption=f"Prediction: {trainset.characters[prediction]} Truth: {trainset.characters[label]}"
                ) for image, label, prediction in zip(images[:examples], predictions[:examples], labels[:examples])]
            })

    def log_failure_cases(count=10):
        with torch.no_grad():
            failure_cases = []
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                for prediction, label, image, output in zip(predictions, labels, images, outputs):
                    if prediction != label:
                        failure_cases += [{
                            "image": image,
                            "prediction": testset.characters[prediction],
                            "label": testset.characters[label],
                            "confidence": output[label]
                        }]

                if total >= len(testset.characters):
                    break
            wandb.log({"failure_cases": [wandb.Image(
                case["image"],
                caption=f"Prediction: {case['prediction']} Truth: {case['label']}"
            ) for case in sorted(failure_cases, key=lambda item: item['confidence'])[:count]]})
            return 100 * correct / total

    log_examples()
    log_failure_cases()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to recognize kanji.")
    parser.add_argument("-o", "--output-path", type=str, default="data/models/recognizer.pt",
                        help="save model to this path")
    parser.add_argument("-i", "--input-path", type=str, default=None,
                        help="load model from this path")
    parser.add_argument("-f", "--fonts-folder", type=str, default="data/fonts",
                        help="path to a folder containing fonts (default: data/fonts)")
    parser.add_argument("-b", "--background-images-folder", type=str, default="data/background-images",
                        help="path to a folder containing background images (default: data/background-images)")
    parser.add_argument("-B", "--batch-size", type=int, default=128,
                        help="the size of the batch used on each training step (default: 128)")
    parser.add_argument("-t", "--training-time", type=float, default=10,
                        help="amount of minutes to train the network (default: 10)")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-3,
                        help="the learning rate of the the optimizer (default: 1e-3)")
    run(parser.parse_args())
