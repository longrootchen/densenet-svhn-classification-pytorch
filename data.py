from torchvision.datasets import ImageFolder


class SVHNDataset(ImageFolder):

    def __init__(self, root, transform=None):
        super(SVHNDataset, self).__init__(root, transform)


if __name__ == '__main__':
    import os
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    root = os.path.join(os.pardir, 'SVHN', 'TrainValid', 'valid')
    ds = SVHNDataset(root, ToTensor())
    len(ds)
    print(ds.class_to_idx)

    dl = DataLoader(ds, batch_size=2, shuffle=True)
    for i, (input_, target) in enumerate(dl):
        print(i)
        print(input_.size())
        print(target)
        print('-' * 40)
        break
