from __future__ import division, print_function

import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm

import dataset as data_util
import networks as net
import networks_masking as mask_net
import networks_masking_lowrank as mask_lowrank_net
import networks1_masking as mask_net1
import utils as utils

import wandb
wandb.init(project="imgnet_sketch1", entity="ucr")

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, if_cuda, model, batch_size, dataset):
        # self.args = args
        self.cuda = if_cuda
        self.model = model
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_path = 'data/%s/train/' % (self.dataset) 
        self.test_path = 'data/%s/test/' % (self.dataset)
        print('Train path: ', self.train_path)
        print('Test path: ', self.test_path)

        # Set up data loader, criterion, and pruner.
        if 'cropped' in self.dataset:
            train_loader = data_util.train_loader_cropped
            test_loader = data_util.test_loader_cropped
        else:
            train_loader = data_util.train_loader
            test_loader = data_util.test_loader
        self.train_data_loader = train_loader(
            self.train_path, self.batch_size, pin_memory=self.cuda)
        self.test_data_loader = test_loader(
            self.test_path, self.batch_size, pin_memory=self.cuda)
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        ts_running_loss = 0

        with torch.no_grad():
            for batch, label in tqdm(self.test_data_loader, desc='Eval'):
                if self.cuda:
                    batch = batch.cuda()
                    label = label.cuda()
                # batch = Variable(batch, volatile=True)

                output = self.model(batch)
                loss = self.criterion(output, label)
                ts_running_loss += loss.item()
                # Init error meter.
                if error_meter is None:
                    topk = [1]
                    if output.size(1) > 5:
                        topk.append(5)
                    error_meter = tnt.meter.ClassErrorMeter(topk=topk)
                error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))

        ts_final_loss = ts_running_loss/len(self.test_data_loader)

        # if 'train_bn' in self.args:
        #     if self.args.train_bn:
        #         self.model.train()
        #     else:
        #         self.model.train_nobn()
        # else:
        #     print('args does not have train_bn flag, probably in eval-only mode.')
        return ts_final_loss, errors

    def do_batch(self, optimizer, batch, label):
        """Runs model for one batch."""
        if self.cuda:
            batch = batch.cuda()
            label = label.cuda()
        batch = Variable(batch)
        label = Variable(label)

        # Set grads to 0.
        # self.model.zero_grad()
        optimizer.zero_grad()
        
        # Do forward-backward.
        output = self.model(batch)
        loss = self.criterion(output, label)
        # print('loss', loss.dtype, loss)
        loss.backward()

        # for param_group in optimizer.param_groups:
        #     print('lr', param_group["lr"])

        ## calc the l2 norm of gradients
        # print('last channel tensor 1', self.model.layer4[2].maskcconv[0].grad.clone())
        # print('last channel tensor 2', self.model.layer4[2].maskcconv[1].grad.clone())
        # print('last channel tensor 3', self.model.layer4[2].maskcconv[2].grad.clone())
        # print('last height tensor 1', self.model.layer4[2].maskhconv[0].grad.clone())
        # print('last height tensor 2', self.model.layer4[2].maskhconv[1].grad.clone())
        # print('last height tensor 3', self.model.layer4[2].maskhconv[2].grad.clone())
        # print('last width tensor 1', self.model.layer4[2].maskwconv[0].grad.clone())
        # print('last width tensor 2', self.model.layer4[2].maskwconv[1].grad.clone())
        # print('last width tensor 3', self.model.layer4[2].maskwconv[2].grad.clone())
        
        grad_first, grad_last = {}, {}
        grad_first['c1'] = torch.norm(self.model.module.layer1[0].mask1.maskcconv.grad.clone())
        grad_first['h1'] = torch.norm(self.model.module.layer1[0].mask1.maskhconv.grad.clone())
        grad_first['w1'] = torch.norm(self.model.module.layer1[0].mask1.maskwconv.grad.clone())
        grad_first['c2'] = torch.norm(self.model.module.layer1[0].mask2.maskcconv.grad.clone())
        grad_first['h2'] = torch.norm(self.model.module.layer1[0].mask2.maskhconv.grad.clone())
        grad_first['w2'] = torch.norm(self.model.module.layer1[0].mask2.maskwconv.grad.clone())
        grad_first['c3'] = torch.norm(self.model.module.layer1[0].mask3.maskcconv.grad.clone())
        grad_first['h3'] = torch.norm(self.model.module.layer1[0].mask3.maskhconv.grad.clone())
        grad_first['w3'] = torch.norm(self.model.module.layer1[0].mask3.maskwconv.grad.clone())

        grad_last['c1'] = torch.norm(self.model.module.layer4[2].mask1.maskcconv.grad.clone())
        grad_last['h1'] = torch.norm(self.model.module.layer4[2].mask1.maskhconv.grad.clone())
        grad_last['w1'] = torch.norm(self.model.module.layer4[2].mask1.maskwconv.grad.clone())
        grad_last['c2'] = torch.norm(self.model.module.layer4[2].mask2.maskcconv.grad.clone())
        grad_last['h2'] = torch.norm(self.model.module.layer4[2].mask2.maskhconv.grad.clone())
        grad_last['w2'] = torch.norm(self.model.module.layer4[2].mask2.maskwconv.grad.clone())
        grad_last['c3'] = torch.norm(self.model.module.layer4[2].mask3.maskcconv.grad.clone())
        grad_last['h3'] = torch.norm(self.model.module.layer4[2].mask3.maskhconv.grad.clone())
        grad_last['w3'] = torch.norm(self.model.module.layer4[2].mask3.maskwconv.grad.clone())

        # # Scale gradients by average weight magnitude.
        # if self.args.mask_scale_gradients != 'none':
        #     for module in self.model.shared.modules():
        #         if 'ElementWise' in str(type(module)):
        #             abs_weights = module.weight.data.abs()
        #             if self.args.mask_scale_gradients == 'average':
        #                 module.mask_real.grad.data.div_(abs_weights.mean())
        #             elif self.args.mask_scale_gradients == 'individual':
        #                 module.mask_real.grad.data.div_(abs_weights)

        # # Set batchnorm grads to 0, if required.
        # if not self.args.train_bn:
        #     for module in self.model.shared.modules():
        #         if 'BatchNorm' in str(type(module)):
        #             if module.weight.grad is not None:
        #                 module.weight.grad.data.fill_(0)
        #             if module.bias.grad is not None:
        #                 module.bias.grad.data.fill_(0)

        # Update params.
        optimizer.step()
        return loss.item(), grad_first, grad_last

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        tr_running_loss = 0
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            loss, grad_first, grad_last = self.do_batch(optimizer, batch, label)
            tr_running_loss+=loss

        tr_final_loss = tr_running_loss/len(self.train_data_loader)
        # if self.args.threshold_fn == 'binarizer':
        #     print('Num 0ed out parameters:')
        #     for idx, module in enumerate(self.model.shared.modules()):
        #         if 'ElementWise' in str(type(module)):
        #             num_zero = module.mask_real.data.lt(5e-3).sum()
        #             total = module.mask_real.data.numel()
        #             print(idx, num_zero, total)
        # elif self.args.threshold_fn == 'ternarizer':
        #     print('Num -1, 0ed out parameters:')
        #     for idx, module in enumerate(self.model.shared.modules()):
        #         if 'ElementWise' in str(type(module)):
        #             num_neg = module.mask_real.data.lt(0).sum()
        #             num_zero = module.mask_real.data.lt(5e-3).sum() - num_neg
        #             total = module.mask_real.data.numel()
        #             print(idx, num_neg, num_zero, total)
        # print('-' * 20)
        return tr_final_loss, grad_first, grad_last

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        ckpt = {
            # 'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'model': self.model.module,
        }

        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, optimizer, scheduler, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []

        if self.cuda:
            self.model = self.model.cuda()

        self.eval()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            # optimizer.update_lr(epoch_idx)
            # if self.args.train_bn:
            self.model.train()
            # else:
            #     self.model.train_nobn()
            tr_loss, grad_first, grad_last = self.do_epoch(epoch_idx, optimizer)
            
            # for i,opt in enumerate(optimizer):
            #     for param_group in opt.param_groups:
            #         print('lr', i, param_group["lr"])
            optimizer.get_lr()
            
            # print(torch.unique(torch.tensor(scheduler.get_last_lr())))
            for sch in scheduler:
                sch.step()
            ts_loss, errors = self.eval()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.

            # log into wandbai
            wandb.log({
                "epoch": epoch_idx,
                "train loss": tr_loss,
                "test loss": ts_loss,
                "accuracy": accuracy,
                "grad channel first1": grad_first['c1'],
                "grad height first1": grad_first['h1'],
                "grad width first1": grad_first['w1'],
                "grad channel first2": grad_first['c2'],
                "grad height first2": grad_first['h2'],
                "grad width first2": grad_first['w2'],
                "grad channel first3": grad_first['c3'],
                "grad height first3": grad_first['h3'],
                "grad width first3": grad_first['w3'],
                "grad channel last1": grad_last['c1'],
                "grad height last1": grad_last['h1'],
                "grad width last1": grad_last['w1'],
                "grad channel last2": grad_last['c2'],
                "grad height last2": grad_last['h2'],
                "grad width last2": grad_last['w2'],
                "grad channel last3": grad_last['c3'],
                "grad height last3": grad_last['h3'],
                "grad width last3": grad_last['w3'],
            })
            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    # 'args': vars(self.args),
                    'training_loss': tr_loss,
                    'test_loss': ts_loss,
                    'accuracy': accuracy,
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        # Make sure masking didn't change any weights.
        # if not self.args.no_mask:
        #     self.check()
        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)

    # def check(self):
    #     """Makes sure that the trained model weights match those of the pretrained model."""
    #     print('Making sure filter weights have not changed.')
    #     if self.args.arch == 'vgg16':
    #         pretrained = net.ModifiedVGG16(original=True)
    #     elif self.args.arch == 'vgg16bn':
    #         pretrained = net.ModifiedVGG16BN(original=True)
    #     elif self.args.arch == 'resnet50':
    #         pretrained = net.ModifiedResNet(original=True)
    #     elif self.args.arch == 'densenet121':
    #         pretrained = net.ModifiedDenseNet(original=True)
    #     elif self.args.arch == 'resnet50_diff':
    #         pretrained = net.ResNetDiffInit(self.args.source, original=True)
    #     else:
    #         raise ValueError('Architecture %s not supported.' %
    #                          (self.args.arch))

    #     for module, module_pretrained in zip(self.model.shared.modules(), pretrained.shared.modules()):
    #         if 'ElementWise' in str(type(module)) or 'BatchNorm' in str(type(module)):
    #             weight = module.weight.data.cpu()
    #             weight_pretrained = module_pretrained.weight.data.cpu()
    #             # Using small threshold of 1e-8 for any floating point inconsistencies.
    #             # Note that threshold per element is even smaller as the 1e-8 threshold
    #             # is for sum of absolute differences.
    #             assert (weight - weight_pretrained).abs().sum() < 1e-8, \
    #                 'module %s failed check' % (module)
    #             if module.bias is not None:
    #                 bias = module.bias.data.cpu()
    #                 bias_pretrained = module_pretrained.bias.data.cpu()
    #                 assert (bias - bias_pretrained).abs().sum() < 1e-8
    #             if 'BatchNorm' in str(type(module)):
    #                 rm = module.running_mean.cpu()
    #                 rm_pretrained = module_pretrained.running_mean.cpu()
    #                 assert (rm - rm_pretrained).abs().sum() < 1e-8
    #                 rv = module.running_var.cpu()
    #                 rv_pretrained = module_pretrained.running_var.cpu()
    #                 assert (rv - rv_pretrained).abs().sum() < 1e-8
    #     print('Passed checks...')


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

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = utils.step_lr(
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
        for i,op in enumerate(self.optimizers):
            # for param_group in op.param_groups:
            print(i, 'lr', op.param_groups[0]['lr'])

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Do stuff."""
    # args = FLAGS.parse_args()

    # Set default train and test path if not provided as input.
    # utils.set_dataset_paths(args)

    # Load the required model.
    # if args.arch == 'vgg16':
    #     model = net.ModifiedVGG16(mask_init=args.mask_init,
    #                               mask_scale=args.mask_scale,
    #                               threshold_fn=args.threshold_fn,
    #                               original=args.no_mask)
    # elif args.arch == 'vgg16bn':
    #     model = net.ModifiedVGG16BN(mask_init=args.mask_init,
    #                                 mask_scale=args.mask_scale,
    #                                 threshold_fn=args.threshold_fn,
    #                                 original=args.no_mask)
    # elif args.arch == 'resnet50':
    #     model = net.ModifiedResNet(mask_init=args.mask_init,
    #                                mask_scale=args.mask_scale,
    #                                threshold_fn=args.threshold_fn,
    #                                original=args.no_mask)
    # elif args.arch == 'densenet121':
    #     model = net.ModifiedDenseNet(mask_init=args.mask_init,
    #                                  mask_scale=args.mask_scale,
    #                                  threshold_fn=args.threshold_fn,
    #                                  original=args.no_mask)
    # elif args.arch == 'resnet50_diff':
    #     assert args.source
    #     model = net.ResNetDiffInit(args.source,
    #                                mask_init=args.mask_init,
    #                                mask_scale=args.mask_scale,
    #                                threshold_fn=args.threshold_fn,
    #                                original=args.no_mask)
    # else:
    #     raise ValueError('Architecture %s not supported.' % (args.arch))
    NUM_OUTPUTS = {}
    NUM_OUTPUTS["imagenet"]=1000
    NUM_OUTPUTS["places"]=365
    NUM_OUTPUTS["stanford_cars_cropped"]=196
    NUM_OUTPUTS["cubs_cropped"]=200
    NUM_OUTPUTS["flowers"]=102
    NUM_OUTPUTS["wikiart"]=195
    NUM_OUTPUTS["sketches"]=250
   
    if_cuda = True
    dataset = 'cubs_cropped'
    batch_size = 32
    lr=5e-3
    lr_mask=5e-3
    weight_decay = 0
    lr_decay_every = 15
    finetune_epochs = 30
    save_name = 'checkpoints/' + dataset + 'clf&msk1_afterbn_mlr_5e-3_wd0_bothcosinesch'
    num_outputs = NUM_OUTPUTS[dataset]
    print('number of output layer and dataset: ', num_outputs, dataset)

    learning = 'maskandclassifier' ## 'classifier', 'maskandclassifier', 'mask


    if learning=='maskandclassifier' or learning=='mask':
        load_name = dataset+'feature_extractor'
        # model = mask_net.resnet50(num_outputs, pretrained=True, progress=True)
        # model = mask_net1.resnet50(num_outputs, pretrained=True, progress=True)
        model = mask_lowrank_net.resnet50(num_outputs, pretrained=True, progress=True)
        loaded_model = torch.load('checkpoints/'+load_name, map_location=torch.device('cpu'))['model'].state_dict()
        model.load_state_dict(loaded_model, strict=False)
        for name, param in model.named_parameters():
            if 'bnmask' in name:
                update_name = name.replace('bnmask', 'bn')
                param.data = loaded_model[update_name].data
    else:
        model = net.resnet50(num_outputs, pretrained=True, progress=True)

    # Add and set the model dataset.
    # model.add_dataset(args.dataset, args.num_outputs)
    # model.set_dataset(args.dataset)
    if if_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()

    # # Initialize with weight based method, if necessary.
    # if not args.no_mask and args.mask_init == 'weight_based_1s':
    #     print('Are you sure you want to try this?')
    #     assert args.mask_scale_gradients == 'none'
    #     assert not args.mask_scale
    #     for idx, module in enumerate(model.shared.modules()):
    #         if 'ElementWise' in str(type(module)):
    #             weight_scale = module.weight.data.abs().mean()
    #             module.mask_real.data.fill_(weight_scale)

    # Create the manager object.
    manager = Manager(if_cuda, model, batch_size, dataset)

    # Perform necessary mode operations.
    mode = 'finetune'
    if mode == 'finetune':
        # if args.no_mask:
        # No masking will be done, used to run baselines of
        # Classifier-Only and Individual Networks.
        # Checks.
        # assert args.lr and args.lr_decay_every
        # assert not args.lr_mask and not args.lr_mask_decay_every
        # assert not args.lr_classifier and not args.lr_classifier_decay_every
        print('running baselines.')

        # Get optimizer with correct params.
        # if args.finetune_layers == 'all':
        if learning=='classifier':
            for name, param in model.named_parameters():
                if 'fc.' not in name:
                    param.requires_grad = False
            params_to_optimize = model.parameters()    
        elif learning=='maskandclassifier':
            mask_params_to_optimize, params_to_optimize = [],[]
            for name, param in model.named_parameters():
                if 'fc.' not in name and 'mask' not in name:  # and 'bn' not in name
                    param.requires_grad = False
                if param.requires_grad:
                    if 'mask' in name:
                        mask_params_to_optimize.append({'params': param, 'lr': lr_mask, 'weight_decay': weight_decay})
                    else:
                        params_to_optimize.append({'params': param, 'lr': lr})
            # params_to_optimize = model.parameters()
        elif learning=='mask':
            for name, param in model.named_parameters():
                if 'mask' not in name:
                    param.requires_grad = False
            params_to_optimize = model.parameters()
        else:
            params_to_optimize = model.parameters()

        num_param = count_parameters(model)
        print('Total number of parameters: ', num_param)
        
        # elif args.finetune_layers == 'classifier':
        #     for param in model.shared.parameters():
        #         param.requires_grad = False
        #     params_to_optimize = model.classifier.parameters()

        optimizer = optim.Adam(params_to_optimize, lr=lr) 
        optimizer_mask = optim.Adam(mask_params_to_optimize, lr=lr_mask)
        # optimizer = optim.SGD(params_to_optimize, lr=lr,
        #                         momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)
        scheduler_mask = optim.lr_scheduler.CosineAnnealingLR(optimizer_mask, finetune_epochs)
        schedulers = [scheduler, scheduler_mask]
        # optimizers = Optimizers()
        # optimizers.add(optimizer, lr, lr_decay_every)
        optimizers = MultipleOptimizer(optimizer, optimizer_mask)
        manager.train(finetune_epochs, optimizers, schedulers,
                        save=True, savename=save_name)

    elif mode == 'eval':
        # Just run the model on the eval set.
        manager.eval()
    # elif args.mode == 'check':
    #     manager.check()


if __name__ == '__main__':
    main()