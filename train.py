import numpy as np
from helpers import make_data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def make_batch(batch_size):
    # this model can only train on data where a spaceship is guaranteed, this is not true when testing
    imgs, labels = zip(*[make_data(has_spaceship=True) for _ in range(batch_size)])
    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return imgs, labels


def main(params):
    # create model
    model = SpaceshipDetector()

    # setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # init train
    model.train()

    # construct optimizer
    opt = optim.Adam(model.parameters(), lr=params['lr'])

    # [train] localization behavior
    for epoch in range(params['epochs']):
        for step in tqdm(range(params['steps_per_epoch']), desc=f'Epoch {epoch}'):

            # get batch
            imgs, labels = make_batch(params['batch_size'])

            # move training data to torch device
            imgs = imgs.to(device)
            labels = labels.to(device)

            # run forward pass on data
            pred = model(imgs)

            # compute loss

            # update weights
            opt.zero_grad() # resetting gradients
            loss.backward() # backwards pass
            opt.step()




if __name__ == "__main__":
    # params config
    params = {'lr': 0.0001,
              'steps_per_epoch': 500,
              'batch_size': 64,
              'epochs': 500}
    main(params)
