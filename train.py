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
from math import ceil
from torchsummary import summary

def make_batch(batch_size, has_spaceship=True):
    # this model can only train on data where a spaceship is guaranteed, this is not true when testing
    imgs, labels = zip(*[make_data(has_spaceship=has_spaceship) for _ in range(batch_size)])
    imgs = torch.unsqueeze(torch.from_numpy(np.asarray(np.stack(imgs), dtype=np.float32)), 1)
    #labels = torch.from_numpy(np.asarray(encode(np.stack(labels)), dtype=np.float32))
    labels = torch.from_numpy(np.asarray(np.apply_along_axis(encode, 1, np.stack(labels)), dtype=np.float32))
    return imgs, labels


def main(params):
    # create model
    model = SpaceshipDetector()

    # cuda setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)

    # print model summary
    summary(model, (1, 200, 200))

    # tensorboard setup
    tb = SummaryWriter(f'runs/{params["name"]}')

    # init train
    model.train()

    # construct optimizer
    opt = optim.Adam(model.parameters(), lr=params['lr'])

    # construct learning rate scheduler
    scheduler = MultiStepLR(opt, milestones=[120, 180], gamma=0.1)

    # [train] localization behavior
    for epoch in range(params['epochs']):
        totalLoss = 0
        totalIOU = 0
        totalOBJ = 0

        with tqdm(range(params['steps_per_epoch']), desc=f'Epoch {epoch}', unit="batch") as steps:
            for step in steps:
                # get batch
                imgs, labels = make_batch(params['batch_size'], has_spaceship=(None if epoch >= params['epoch_threshold'] else True))

                # move training data to torch device
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # run forward pass on data
                pred = model(imgs)

                ### CUSTOM LOSS COMPUTATION ###
                # compute norm loss for bbox encodings
                l2 = nn.MSELoss(reduction='none') # set reduce to 'none' so we can apply (negative) classification mask
                l1 = nn.L1Loss(reduction='none')  # set reduce to 'none' so we can apply (negative) classification mask

                # construct mask vector based on negative samples
                mask = torch.squeeze(labels[:,-1:]) > 0

                # compute raw distance loss (L2 / L1)
                loss = l2(pred[:,:-1], labels[:,:-1]) + l1(pred[:,:-1], labels[:,:-1])

                # reduce loss via summing label loss (unraveled MSE)
                loss = torch.sum(loss, 1)

                # apply negative sample mask to the distance loss
                loss = torch.masked_select(loss, mask)

                # reduce loss tensor into scalar via mean
                if len(loss) > 0:
                    loss = torch.mean(loss) / 6
                else:
                    loss = 0

                if epoch >= params['epoch_threshold']:
                    # freeze feature extractor (cnn) weights so that we don't hurt localization
                    for p in model.featureExtractor.parameters():
                        p.requires_grad = False

                    # compute L2 loss for classification
                    cl = nn.MSELoss()
                    loss += cl(pred[:,-1:], labels[:,-1:]) / 10

                # decode latent representation into raw labels
                decoded_pred = decode(pred.cpu().detach().numpy())
                decoded_true = decode(labels.cpu().detach().numpy())
                
                # compute IOU metric
                iou = np.mean([score_iou(decoded_true[i][:-1], decoded_pred[i][:-1]) for i in range(len(decoded_true)) if not (None == score_iou(decoded_true[i][:-1], decoded_pred[i][:-1]))])

                # average IOU over samples in batch
                iou = np.mean(iou)

                # compute objectness correctness metric
                binClassAcc = lambda x: bool(ceil(x)) if not (x is None) else None
                obj = np.array([(binClassAcc(score_iou(decoded_true[i][:-1], decoded_pred[i][:-1])) in {None, True}) for i in range(len(decoded_true))], dtype=int)

                # avg objectness over samples in batch (accuracy)
                obj = np.mean(obj)

                # update weights
                opt.zero_grad() # resetting gradients
                loss.backward() # backwards pass
                opt.step()

                totalLoss += loss.item()
                totalIOU  += iou
                totalOBJ  += obj

                # update progress bar
                steps.set_postfix(loss=totalLoss / (step+1), iou=totalIOU / (step+1), obj=totalOBJ / (step+1))

            # update learning rate scheduler
            scheduler.step()

            # update tensorboard
            tb.add_scalar('loss/train', totalLoss / params['steps_per_epoch'], epoch)
            tb.add_scalar('iou/train', totalIOU / params['steps_per_epoch'], epoch)
            tb.add_scalar('obj/train', totalOBJ / params['steps_per_epoch'], epoch)

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
    params = {'name': 'hydra',
              'path': 'zoo',
              'lr': 0.001,
              'steps_per_epoch': 250,
              'batch_size': 64,
              'epochs': 500,
              'epoch_threshold': 150}
    main(params)
