from __future__ import division, print_function

import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm

import dataset as data_util

from resnet import resnet50
from model_summary import summary

import wandb
import random

from resnet_with_dense_mask import resnet50with_dense_mask
from resnet_with_dense_mask_after_conv import resnet50with_dense_mask_after_conv

# wandb.init(project="My-Imagenet-Sketch", entity="kaykobad")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Manager(object):
    def __init__(self, model, batch_size, dataset):
        self.model = model
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_path = 'data/imagenet-sketch/%s/train/' % (self.dataset)
        self.test_path = 'data/imagenet-sketch/%s/test/' % (self.dataset)
        print('Train path: ', self.train_path)
        print('Test path: ', self.test_path)

        # Set up data loader, criterion, and pruner.
        if 'cropped' in self.dataset:
            train_loader = data_util.train_loader_cropped
            test_loader = data_util.test_loader_cropped
        else:
            train_loader = data_util.train_loader
            test_loader = data_util.test_loader
        self.train_data_loader = train_loader(self.train_path, self.batch_size, pin_memory=True)
        self.test_data_loader = test_loader(self.test_path, self.batch_size, pin_memory=True)
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        ts_running_loss = 0

        with torch.no_grad():
            for batch, label in tqdm(self.test_data_loader, desc='Eval'):
                batch = batch.to(device)
                label = label.to(device)

                output = self.model(batch)
                loss = self.criterion(output, label)
                ts_running_loss += loss.item()

                if error_meter is None:
                    topk = [1]
                    if output.size(1) > 5:
                        topk.append(5)
                    error_meter = tnt.meter.ClassErrorMeter(topk=topk)
                error_meter.add(output.data, label)

                # Detaching values
                batch.detach()
                label.detach()
                output.detach()

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))

        ts_final_loss = ts_running_loss / len(self.test_data_loader)

        return ts_final_loss, errors

    def do_batch(self, optimizer, batch, label):
        batch = batch.to(device)
        label = label.to(device)
        batch = Variable(batch)
        label = Variable(label)

        optimizer.zero_grad()

        # Do forward-backward.
        output = self.model(batch)
        loss = self.criterion(output, label)
        loss.backward()

        # Detaching values
        loss.detach()
        batch.detach()
        label.detach()
        output.detach()
        del batch
        del label
        del output

        optimizer.step()
        return loss.item()

    def do_epoch(self, epoch_idx, optimizer):
        tr_running_loss = 0
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            loss = self.do_batch(optimizer, batch, label)
            tr_running_loss += loss
            del loss

        tr_final_loss = tr_running_loss / len(self.train_data_loader)

        return tr_final_loss

    def save_model(self, epoch, best_accuracy, errors, savename):
        ckpt = {
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'model': self.model,
        }

        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, optimizer, scheduler, save=True, savename='', best_accuracy=0):
        best_accuracy = best_accuracy
        error_history = []

        self.model = self.model.to(device)

        self.eval()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            self.model.train()

            tr_loss = self.do_epoch(epoch_idx, optimizer)

            optimizer.get_lr()

            for sch in scheduler:
                sch.step()
            ts_loss, errors = self.eval()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.

            wandb.log({
                "epoch": epoch_idx,
                "train loss": tr_loss,
                "test loss": ts_loss,
                "accuracy": accuracy,
            })

            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'training_loss': tr_loss,
                    'test_loss': ts_loss,
                    'accuracy': accuracy,
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' % (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

                # Print Mask
                # print("Last mask:", self.model.module.layer4[2].mask3.mask.view(1, 1, 1, -1))
                # print("Last mask norm:", torch.norm(self.model.module.layer4[2].mask3.mask.data))
                # print("Last mask:", self.model.module.layer4[2].mask3.mask.data)
                # # print("First mask:", self.model.module.layer1[0].mask1.mask.view(1, 1, 1, -1))
                # print("First mask norm:", torch.norm(self.model.module.layer1[0].mask1.mask.data))
                # print("First mask:", self.model.module.layer1[0].mask1.mask.data)

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' % (100 - best_accuracy, best_accuracy))
        print('-' * 16)


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, lr_decay_factor=0.1):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.lr_decay_factor = lr_decay_factor

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def step_lr(self, epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
        """Handles step decay of learning rate."""
        factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
        new_lr = base_lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print('Set lr to ', new_lr)
        return optimizer

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = self.step_lr(
                epoch_idx, init_lr, decay_every,
                self.lr_decay_factor, optimizer)


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def get_lr(self):
        for i, op in enumerate(self.optimizers):
            # for param_group in op.param_groups:
            print(i, 'lr', op.param_groups[0]['lr'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    wandb.init(project="My-Imagenet-Sketch", entity="kaykobad", name=wandb_name)
    print('number of output layer and dataset: ', num_outputs, dataset)

    if dense_2d_mask:
        model = resnet50with_dense_mask_after_conv(num_outputs, pretrained=True)
    elif dense_mask:
        model = resnet50with_dense_mask(num_outputs, pretrained=True, use_masks=use_dense_masks)
    else:
        model = resnet50(num_outputs, pretrained=True, use_masks=use_masks, mask_rank=mask_rank)
    model = nn.DataParallel(model)
    model = model.to(device)

    # TODO: Tune it properly
    for name, param in model.named_parameters():
        if 'fc.' not in name and 'mask' not in name and 'bn' not in name:
            param.requires_grad = False
        # if 'fc.' not in name:
        #     param.requires_grad = False
        # if (optimize_bn and 'bn' in name) or ((True in use_masks) and 'mask' in name) or (dense_mask and (True in use_dense_masks) and 'mask' in name) or dense_2d_mask:
        #     param.requires_grad = True

    num_param = count_parameters(model)
    print('Total number of parameters: ', num_param)
    params_to_optimize = model.parameters()
    optimizer = optim.Adam(params_to_optimize, lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    schedulers = [scheduler]
    optimizers = MultipleOptimizer(optimizer)

    print("Running on: ", device)
    print("Model summary:")
    total_params, trainable_params = summary(model, (3, 224, 224))
    print("Total params:", total_params, "Trainable params:", trainable_params)

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(model)

    manager = Manager(model, batch_size, dataset)
    manager.train(finetune_epochs, optimizers, schedulers, save=True, savename=save_name)


if __name__ == '__main__':
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    NUM_OUTPUTS = {
        "imagenet": 1000,
        "places": 365,
        "stanford_cars_cropped": 196,
        "cubs_cropped": 200,
        "flowers": 102,
        "wikiart": 195,
        "sketches": 250,
    }

    # Parameters
    use_masks = [False, False, False, False]
    mask_rank = 5
    dataset = 'wikiart'
    checkpoint_suffix = '_fc+bn'
    batch_size = 32
    lr = 5e-3
    finetune_epochs = 30
    save_name = 'checkpoints/' + dataset + checkpoint_suffix + '.pth'
    num_outputs = NUM_OUTPUTS[dataset]
    optimize_bn = True
    wandb_name = 'Wikiart-FC+BN'
    dense_mask = False
    dense_2d_mask = False
    use_dense_masks = [False, False, False]

    # Setting the seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    main()
