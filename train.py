import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from adapter import Adapter
from metrics import ConfusionCounter as Counter

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

        self.adapter = Adapter(args, self.optimizer)

        self.half_acc = args.half_acc
        self.n_classes = args.n_classes

        self.grad_clip_norm = args.grad_clip_norm
        self.grad_scale_factor = args.grad_scale_factor

        self.criterion = nn.__dict__[args.criterion + 'Loss'](reduction = 'mean').cuda()


    def train(self, epoch, data_loader, device):
        self.model.train()
        n_batches = len(data_loader)
        self.adapter.schedule(self.writer.state)

        for i, (images, labels) in enumerate(data_loader):

            images = images.to(device)
            labels = labels.to(device, dtype = torch.long)

            logits = self.model(images)

            loss = self.criterion(logits, labels)

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


    def eval(self, test_loader, device):
        self.model.eval()
        counter = Counter(self.n_classes)

        for images, labels in test_loader:

            images = images.to(device)

            with torch.no_grad():
                logits = self.model(images)

            predictions = outputs.detach().cpu().numpy().max(dim = 1)[1]

            metrics.update(labels, predictions)

        return counter.to_metrics()


    def get_model(self):
        return self.model.module if torch.typename(model).find('DataParallel') != -1 else self.model
