import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import mat_utils
import utils
import queue

from torch.autograd import Variable
from builtins import zip as xzip


class Trainer:

    def __init__(self, args, model, writer):
        self.model = model
        self.writer = writer

        self.list_params = list(model.parameters())

        if args.half_acc:
            self.copy_params = [param.clone().detach() for param in self.list_params]
            self.model = self.model.half()

            for param in self.copy_params:
                param.requires_grad = True
                param.grad = param.data.new_zeros(param.size())

            self.optimizer = optim.Adam(self.copy_params, args.learn_rate, weight_decay = args.weight_decay)
        else:
            self.optimizer = optim.Adam(self.list_params, args.learn_rate, weight_decay = args.weight_decay)

        self.n_epochs = args.n_epochs
        self.n_iters_warmup = args.n_iters_warmup
        self.n_iters_check_loss = args.n_iters_check_loss
        self.n_iters_check_model = args.n_iters_check_model
        self.half_acc = args.half_acc

        self.learn_rate = args.learn_rate
        self.grad_clip_norm = args.grad_clip_norm
        self.grad_scale_factor = args.grad_scale_factor

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean').cuda()

    def train(self, epoch, data_loader, device):
        self.model.train()
        self.adapt_learn_rate(epoch)

        n_batches = len(data_loader)

        batch_loss = list()

        for i, (images, labels) in enumerate(data_loader):

            images = images.to(device)
            labels = labels.to(device)

            predictions = self.model(images)

            loss = self.criterion(predictions, labels)

            batch_size.append(images.size(0))
            batch_loss.append(loss)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
            self.optimizer.step()

            disp_loss = np.sum(batch_loss[max(i - self.n_iters_check_loss + 1, 0):i + 1]) / min(i + 1, self.n_iters_check_loss)

            print('| train Epoch[%d] [%d/%d] | Loss %1.4f' % (epoch, i, n_batches, disp_loss))

        print('\n=> | train Epoch[%d] finishes | Loss: %1.4f <=\n' % (epoch, np.sum(batch_loss) / n_batches))

        writer.integrate_loss(n_batches, batch_loss)

    def adapt_learn_rate(self, epoch):
        if epoch - 1 < self.num_epochs * 0.6:
            learn_rate = self.learn_rate
        elif epoch - 1 < self.num_epochs * 0.9:
            learn_rate = self.learn_rate * 0.2
        else:
            learn_rate = self.learn_rate * 0.04

        if self.do_track and epoch != 1:
            learn_rate /= 2

        for group in self.optimizer.param_groups:
            group['lr'] = learn_rate
