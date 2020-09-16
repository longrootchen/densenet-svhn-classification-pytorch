import os
import warnings

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from data import SVHNDataset
from utils import Evaluator
from configs import Config

warnings.filterwarnings('ignore')

cls_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


def test(device, model, test_loader, vis_conf_mat=False):
    evaluator = Evaluator(Config.num_classes)

    model.eval()
    with tqdm(test_loader) as pbar:
        pbar.set_description('Eval in test set')

        for i, (input_, target) in enumerate(test_loader):
            input_ = torch.tensor(input_, device=device, dtype=torch.float32)
            target = torch.tensor(target, device=device, dtype=torch.long)

            with torch.no_grad():
                output = model(input_)

            true = target.cpu().numpy()
            pred = output.max(dim=1)[1].cpu().numpy()
            evaluator.update_matrix(true, pred)

            pbar.update()

    if vis_conf_mat:
        evaluator.show_matrix(cls_to_idx, save_matrix=True)

    return evaluator.error()


if __name__ == '__main__':
    # ========== config ==========
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========== get model ==========
    model = models.__dict__[Config.arch]()

    cp_path = os.path.join(os.curdir, 'checkpoints', 'last_checkpoint.pth')
    checkpoint = torch.load(cp_path, map_location=device)

    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # ========== get data ==========
    data_root = os.path.join(os.pardir, 'SVHN', 'CroppedDigits', 'test')
    test_set = SVHNDataset(data_root, transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    # ========== eval in test set ==========
    err = test(device, model, test_loader, vis_conf_mat=True)
    print(err)
