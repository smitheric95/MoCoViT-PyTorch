"""Script to train MoCoViT on ImageNet.

Train and val .tar files can be downloaded from https://image-net.org
and should be placed in ./imagenet.

Based loosely on https://github.com/pytorch/examples/blob/main/imagenet/main.py
"""
import argparse
from distutils.util import strtobool
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from mocovit import MoCoViT

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L464
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MoCoViT model on an ImageNet-1k dataset.')
    parser.add_argument('--imagenet_path', type=str, default='./imagenet', help="Path to ImageNet-1k directory containing 'train' and 'val' folders. Default './imagenet'.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use for training. Default 0.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for which to train. Default 20.')
    parser.add_argument('--validate', choices=('True', 'False'), default='True', help='If True, run validation after each epoch. Default True.')
    parser.add_argument('--train_batch', type=int, default=128, help='Batch size to use for training. Default 128.')
    parser.add_argument('--val_batch', type=int, default=1024, help='Batch size to use for validation. Default 1024.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use while loading dataset splits. Default 4.')
    args = parser.parse_args()
    args.validate = strtobool(args.validate)

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

    # define train and val datasets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = torchvision.datasets.ImageNet(args.imagenet_path, split='train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=0)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    val_dataset = torchvision.datasets.ImageNet(args.imagenet_path, split='val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, num_workers=0)

    # load network onto gpu
    model = MoCoViT()
    model.train()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()

    print('Starting Training...')
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
                running_loss = 0.0

        # save checkpoint
        torch.save(model.state_dict(), './checkpoints/epoch%s.pt' % (epoch))

        # validate
        if args.validate:
            model.eval()

            all_outputs = torch.empty((len(val_dataset), 1000))
            all_labels = torch.empty(len(val_dataset))

            print('\nValidating...')
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    num_labels = len(labels)
                    outputs = model(inputs)

                    offset = i * args.val_batch
                    all_outputs[offset:(offset)+num_labels] = outputs
                    all_labels[offset:(offset)+num_labels] = labels

            acc1, acc5 = accuracy(all_outputs, all_labels, topk=(1, 5))
            print("Overall  Acc@1: %.5f, Acc@5: %.5f\n" % (acc1, acc5))

    print('\nFinished Training.')
