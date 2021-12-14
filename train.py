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

        self.enc_crop = args.enc_crop
        self.attention = args.attention
        self.n_classes = args.n_classes

        self.grad_clip_norm = args.grad_clip_norm


    def set_loader(self, args, data_loader):
        self.data_loader = data_loader

        self.crop_func = RandomCrop(args.crop_rate)
        self.optimizer = optim.Adam(self.model.parameters(), args.learn_rate, weight_decay = args.weight_decay)

        self.adapter = adapter.__dict__.get(args.adapter + 'Adapter')
        self.adapter = self.adapter(args, self.optimizer, len(data_loader))

        self.criterion = nn.__dict__.get(args.criterion + 'Loss', criteria.__dict__.get(args.criterion + 'Loss'))
        self.criterion = self.criterion(reduction = 'none', ignore_index = args.n_classes).cuda()


    def set_test_loader(self, test_loader):
        self.test_loader = test_loader


    def train(self, device):
        self.model.train()
        self.adapter.schedule(self.writer.state)

        for i, batch in enumerate(self.data_loader):
            if self.enc_crop:
                batch['image'], batch['label'] = self.crop_func(batch['image'], batch['label'])

            batch['image'] = batch['image'].to(device)
            batch['label'] = batch['label'].to(device, dtype = torch.long)
            if self.attention:
                batch['atten'] = batch['atten'].to(device)

            logits = self.model(batch['image'])
            loss = self.criterion(logits['layer4'], batch['label'])
            if 'layer3' in logits:
                loss += self.criterion(logits['layer3'], batch['label'])

            loss = torch.mul(loss, batch['atten']).mean() if self.attention else loss.mean()

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            self.writer.inc_iter(loss.item())
            self.writer.print_iter(i, self.adapter.n_batches)
            self.writer.check_model(self)

            self.adapter.schedule(self.writer.state)

        self.writer.save_records()
        self.writer.print_epoch(self.adapter.n_batches)
        self.writer.inc_epoch()


    def eval(self, device, save_spec = False):
        self.model.eval()
        print('=> => begins a evaluation round')
        counter = Counter(self.n_classes)

        n_batches = len(self.test_loader)

        for i, batch in enumerate(self.test_loader):
            batch['image'] = batch['image'].to(device)

            with torch.no_grad():
                logits = self.model(batch['image'])

            batch['label'] = batch['label'].numpy()
            predictions = logits['layer4'].detach().cpu().numpy().argmax(axis = 1)

            if save_spec:
                self.writer.save_spec(predictions, i)

            counter.update(batch['label'], predictions)

            print('=> => => | test Batch [{:d}:{:d}] |'.format(i + 1, n_batches))

        print('<= <= evaluation round finishes')
        self.model.train()
        return counter.to_metrics()


    def get_model(self):
        return self.model.module if torch.typename(self.model).find('DataParallel') != -1 else self.model
