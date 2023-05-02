"""Script to test MoCoViT against ImageNet.

Test .tar files can be downloaded from https://image-net.org
and should be placed in ./imagenet.
"""
import os
import argparse
from distutils.util import strtobool
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
from mocovit import MoCoViT

from mapping import get_mapping
mapping = get_mapping()

class TestDataset(Dataset):
    """Dataset just for loading ImageNet test split"""
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs = sorted([img for img in os.listdir(img_dir) if 'ILSVRC2012_test' in img and img.endswith('JPEG')])
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        tensor_img = self.transforms(img)

        return tensor_img, img_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a MoCoViT model against an ImageNet-1k dataset.')
    parser.add_argument('--imagenet_path', type=str, default='./imagenet', help="Path to ImageNet-1k directory containing 'test' folder. Default './imagenet'.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use for testing. Default 0.')
    parser.add_argument('--epoch', type=int, default=-1, help="Epoch of model to use for testing. Default './checkpoints/default.pt'")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use while loading dataset splits. Default 4.')
    parser.add_argument('--verbose', choices=('True', 'False'), default='False', help='If True, print prediction information. Default False.')
    args = parser.parse_args()
    args.verbose = strtobool(args.verbose)

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

    test_dataset = TestDataset(os.path.join(args.imagenet_path, 'test'))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    num_imgs = len(test_dataset)

    # load model
    checkpoint_name = 'epoch%s.pt' % args.epoch if args.epoch >= 0 else 'default.pt'
    model = MoCoViT()
    model.load_state_dict(torch.load(os.path.join('./checkpoints', checkpoint_name)))
    print('Loaded model checkpoint %s.' % checkpoint_name)

    # set model to eval mode, store on device
    model.eval()
    model.to(device)

    print('\nRunning against test data...')
    output_file = open('output_epoch%s.txt' % args.epoch,'w')
    for i, data in enumerate(test_loader):
        # load image batch
        inputs = data[0].to(device)

        # forward pass
        with torch.no_grad():
            result = model(inputs)
        
        results = " ".join([str(r.item()) for r in torch.topk(result.flatten(), 5).indices])
        if args.verbose:
            preds = [mapping[r.item()] for r in torch.topk(result.flatten(), 5).indices]
            print(data[1][0] + ": " + str(preds))

        output_file.writelines(results + "\n")

        if i % (num_imgs * 0.01) == 0:
            print("Tested image %s of %s." % (i, num_imgs))
    
    print('Testing complete.')
    output_file.close()
