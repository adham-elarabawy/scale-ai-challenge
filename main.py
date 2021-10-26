from helpers import make_data, score_iou
import numpy as np
import torch
from tqdm import tqdm
import os.path
from helpers import decode
from train import make_batch
from model import SpaceshipDetector



def eval(params):
    # config
    VERBOSE = True

    # pytorch device setup
    device = torch.device("cpu")
    if VERBOSE:
        print(f'Using device: {device}')

    # Finding the appropriate epoch checkpoint model path from params
    model_dir_path = os.path.join(params['dir'], params['name'])
    checkpoints = [f for f in os.listdir(model_dir_path) if f.endswith('.pth')]
    checkpoints.sort()

    if params['checkpoint_epoch'] == 'latest':
        model_file = checkpoints[-1]
    else:
        model_file = str(int(params['checkpoint_epoch'])) + '.pth'
        if not (model_file in checkpoints):
            print(f'FATAL: Checkpoint not found at {model_file}.')
            exit()

    # load model
    model_path = os.path.join(model_dir_path, model_file)
    model = SpaceshipDetector()
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    if VERBOSE:
        print(f'Evaluating model [{params["name"]}] at epoch checkpoint [{model_path}].')

    ious = []
    for _ in tqdm(range(params['numIters'])):
        # create test data
        imgs, labels = make_batch(1, has_spaceship=True)
        img = imgs[0]
        label = decode(labels.numpy())[0][:-1]

        # convert np -> tensor and move to gpu/cpu
        img.to(device)

        # run forward pass prediction
        pred = model.forward(img[None])

        # decode network output into native label representation
        pred = decode(pred.detach().cpu().numpy())
        pred = np.squeeze(pred)[:-1]

        # compute iou score
        ious.append(score_iou(label, pred))

    ious = np.asarray(ious, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    print((ious > 0.7).mean())


if __name__ == "__main__":
    # params config
    params = {'name': 'hydra',
              'dir': 'zoo',
              'checkpoint_epoch': 'latest',
              'has_spaceship': True,
              'numIters': 1000}
    eval(params)
