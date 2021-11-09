import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import adapter
import criteria

from metrics import Counter
from augmentation import RandomCrop

class Trainer:

    def __init__(self, args, model, writer):
        self.model = model
        self.writer = writer

        self.aux_loss = args.aux_loss
        self.enc_crop = args.enc_crop
        self.n_classes = args.n_classes

        self.grad_clip_norm = args.grad_clip_norm


    def set_loader(args, data_loader):
        self.data_loader = data_loader

        self.crop_func = RandomCrop(args.crop_rate)
        self.optimizer = optim.Adam(model.parameters(), args.learn_rate, weight_decay = args.weight_decay)

        self.adapter = adapter.__dict__.get(args.adapter + 'Adapter')
        self.adapter = self.adapter(args, self.optimizer, len(data_loader))

        self.criterion = nn.__dict__.get(args.criterion + 'Loss', criteria.__dict__.get(args.criterion + 'Loss'))
        self.criterion = self.criterion(ignore_index = args.n_classes).cuda()


    def set_test_loader(test_loader):
        self.test_loader = test_loader


    def train(self, device):
        self.model.train()
        self.adapter.schedule(self.writer.state)

        for i, (images, labels) in enumerate(self.data_loader):
            if self.enc_crop:
                images, labels = self.crop_func(images, labels)

            images = images.to(device)
            labels = labels.to(device, dtype = torch.long)

            logits = self.model(images)
            loss = self.criterion(logits['out'], labels)
            if self.aux_loss:
                loss += self.criterion(logits['aux'], labels)

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.list_params, self.grad_clip_norm)
            self.optimizer.step()

            self.writer.inc_iter(loss.item())
            self.writer.print_iter(i, self.adapter.n_batches)

            self.writer.check_model(self)
            self.adapter.schedule(self.writer.state)

        self.writer.inc_epoch()
        self.writer.save_records()

        self.writer.print_epoch(self.adapter.n_batches)


    def eval(self, device, save_spec = False):
        self.model.eval()
        print('=> => begins a evaluation round')
        counter = Counter(self.n_classes)

        n_batches = len(self.test_loader)

        for i, (images, labels) in enumerate(self.test_loader):
            images = images.to(device)

            with torch.no_grad():
                logits = self.model(images)

            labels = labels.cpu().numpy()
            predictions = logits['out'].detach().cpu().numpy().argmax(axis = 1)

            if save_spec:
                self.writer.save_spec(predictions, i)

            counter.update(labels, predictions)

            print('=> => => | test Batch [{:d}:{:d}] |'.format(i + 1, n_batches))

        print('<= <= evaluation round finishes')
        self.model.train()
        return counter.to_metrics()


    def get_model(self):
        return self.model.module if torch.typename(self.model).find('DataParallel') != -1 else self.model
