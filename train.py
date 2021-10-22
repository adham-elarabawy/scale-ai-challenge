import numpy as np
from helpers import make_data, encode, decode, decode_torch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from metrics.oriented_iou_loss import cal_giou
from model import SpaceshipDetector
import os.path
from torch.utils.tensorboard import SummaryWriter



def make_batch(batch_size):
    # this model can only train on data where a spaceship is guaranteed, this is not true when testing
    imgs, labels = zip(*[make_data(has_spaceship=True) for _ in range(batch_size)])
    imgs = torch.unsqueeze(torch.from_numpy(np.asarray(np.stack(imgs), dtype=np.float32)), 1)
    labels = torch.from_numpy(np.asarray(encode(np.stack(labels)), dtype=np.float32))
    return imgs, labels


def main(params):
    # create model
    model = SpaceshipDetector()

    # cuda setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # tensorboard setup
    tb = SummaryWriter(f'runs/{params["name"]}')

    # init train
    model.train()

    # construct optimizer
    opt = optim.Adam(model.parameters(), lr=params['lr'])

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

                # decode latent representation into raw labels
                decoded_pred = decode_torch(pred)
                decoded_true = decode_torch(labels)

                # clean them up for IOU computation
                decoded_pred = torch.unsqueeze(decoded_pred, 1)
                decoded_true = torch.unsqueeze(decoded_true, 1)

                # compute generalized IOU (GIOU) loss
                giou_loss, iou = cal_giou(decoded_pred, decoded_true)

                # average loss & IOU over samples in batch
                giou_loss = torch.mean(giou_loss)
                iou = torch.mean(iou)

                # update weights
                opt.zero_grad() # resetting gradients
                giou_loss.backward() # backwards pass
                opt.step()

                totalLoss += giou_loss.item()
                totalIOU  += iou

                # update progress bar
                steps.set_postfix(loss=totalLoss / params['steps_per_epoch'], gIOU=totalIOU / params['steps_per_epoch'])

            # update tensorboard
            tb.add_scalar('loss/train', totalLoss / params['steps_per_epoch'], epoch)
            tb.add_scalar('g_iou/train', totalIOU / params['steps_per_epoch'], epoch)

            # save the model at every epoch
            model_path = os.path.join(params['path'], params['name'], str(epoch) + '.pth')
            if not os.path.exists(os.path.join(params['path'], params['name'])): os.makdirs(model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': giou_loss
            }, model_path)



if __name__ == "__main__":
    # params config
    params = {'name': 'mk0',
              'path': 'zoo',
              'lr': 0.0001,
              'steps_per_epoch': 500,
              'batch_size': 64,
              'epochs': 500}
    main(params)
