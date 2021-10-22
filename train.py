import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class Trainer:

    def __init__(self, args, model, writer, adapter):
        self.model = model
        self.writer = writer
        self.adapter = adapter

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

        self.half_acc = args.half_acc

        self.grad_clip_norm = args.grad_clip_norm
        self.grad_scale_factor = args.grad_scale_factor

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean').cuda()


    def train(self, epoch, data_loader, device):
        self.model.train()
        self.adapt_learn_rate(epoch)

        n_batches = len(data_loader)

        for i, (images, labels) in enumerate(data_loader):

            images = images.to(device)
            labels = labels.to(device, dtype = torch.long)

            predictions = self.model(images)

            loss = self.criterion(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.list_params, self.grad_norm)
            self.optimizer.step()

            self.writer.inc_iter(loss.item())
            self.writer.check_model(self)

            print('| train Epoch[{:d}] [{:d}:{:d}]'.format(epoch, i + 1, n_batches), self.writer.get_loss())

            self.adapter.schedule(self.writer.state)

        self.writer.inc_epoch()

        print('\n=> | train Epoch[{:d}}] finishes | Epoch-Mean: {:1.4f} <=\n'.format(epoch, self.writer.get_epoch_mean(n_batches)))

        self.adapter.schedule(self.writer.state)


    def eval(self, test_loader, device):
        self.model.eval()

    def get_model(self):
        return self.model.module if torch.typename(model).find('DataParallel') != -1 else self.model
