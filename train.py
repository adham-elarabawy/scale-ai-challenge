import numpy as np
from helpers import make_data, encode, decode, decode_torch, score_iou
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import SpaceshipDetector
import os.path
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR



def make_batch(batch_size, has_spaceship=True):
    # this model can only train on data where a spaceship is guaranteed, this is not true when testing
    imgs, labels = zip(*[make_data(has_spaceship=has_spaceship) for _ in range(batch_size)])
    imgs = torch.unsqueeze(torch.from_numpy(np.asarray(np.stack(imgs), dtype=np.float32)), 1)
    labels = torch.from_numpy(np.asarray(encode(np.stack(labels)), dtype=np.float32))
    return imgs, labels


def main(params):
    # create model
    model = SpaceshipDetector()

    # cuda setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)

    # tensorboard setup
    tb = SummaryWriter(f'runs/{params["name"]}')

    # init train
    model.train()

    # construct optimizer
    opt = optim.Adam(model.parameters(), lr=params['lr'], eps=1e-07)

    # construct learning rate scheduler
    scheduler = MultiStepLR(opt, milestones=[120, 180], gamma=0.1)

    # [train] localization behavior
    for epoch in range(params['epochs']):
        totalLoss = 0
        totalIOU = 0

        with tqdm(range(params['steps_per_epoch']), desc=f'Epoch {epoch}', unit="batch") as steps:
            for step in steps:
                # get batch
                imgs, labels = make_batch(params['batch_size'])

                # move training data to torch device
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # run forward pass on data
                pred = model(imgs)

                # compute loss
                l2 = nn.MSELoss()
                l1 = nn.L1Loss()
                loss = l2(pred, labels) + l1(pred, labels)

                # decode latent representation into raw labels
                if device.type == 'cpu':
                    decoded_pred = decode(pred.detach().numpy())
                    decoded_true = decode(labels.detach().numpy())
                else:
                    decoded_pred = decode(pred.cpu().detach().numpy())
                    decoded_true = decode(labels.cpu().detach().numpy())
                
                # compute generalized IOU (GIOU) loss
                iou = np.mean([score_iou(decoded_true[i], decoded_pred[i]) for i in range(len(decoded_true))])

                # average IOU over samples in batch
                iou = np.mean(iou)

                # update weights
                opt.zero_grad() # resetting gradients
                loss.backward() # backwards pass
                opt.step()

                totalLoss += loss.item()
                totalIOU  += iou

                # update progress bar
                steps.set_postfix(loss=loss.item(), iou=iou)

            # update learning rate scheduler
            scheduler.step()

            # update tensorboard
            tb.add_scalar('loss/train', totalLoss / params['steps_per_epoch'], epoch)
            tb.add_scalar('iou/train', totalIOU / params['steps_per_epoch'], epoch)

            # save the model at every epoch
            model_path = os.path.join(params['path'], params['name'], str(epoch) + '.pth')
            if not os.path.exists(os.path.join(params['path'], params['name'])): os.makedirs(os.path.join(params['path'], params['name']))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, model_path)


if __name__ == "__main__":
    # params config
    params = {'name': '2',
              'path': 'zoo',
              'lr': 0.001,
              'steps_per_epoch': 500,
              'batch_size': 64,
              'epochs': 500}
    main(params)
