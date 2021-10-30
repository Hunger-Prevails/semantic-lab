import os
import json
import torch
import numpy as np

class Writer:

    def __init__(self, args, state, test_loader):
        self.n_iters_check_loss = args.n_iters_check_loss
        self.n_iters_check_model = args.n_iters_check_model

        self.state = state if state else dict(past_epochs = 0, past_iters = 0)

        self.test_loader = test_loader

        self.save_path = os.path.join(args.save_path, args.head + '_' + args.backbone + '_' + args.suffix)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.config_path = os.path.join(self.save_path, 'configs.json')
        with open(self.config_path, 'w') as file:
            file.write(json.dumps(vars(args)))

        self.record_path = os.path.join(self.save_path, 'records.json')
        if args.resume:
            assert os.path.exists(self.record_path)

            with open(self.record_path, 'r') as file:
                self.records = self.clamp_to_state(json.load(file))
        else:
            self.records = dict(losses = list(), metrics = dict())


    def inc_iter(self, loss):
        self.records['losses'].append(loss)
        self.state['past_iters'] += 1


    def inc_epoch(self):
        self.state['past_epochs'] += 1


    def clamp_to_state(self, records):
        self.records = dict()
        self.records['losses'] = records['losses'][:self.state['past_iters']]
        self.records['metrics'] = dict([(key, records['metrics'][key]) for key in records['metrics'] if key <= self.state['past_iters']])


    def get_loss(self):
        window_sum = np.sum(self.records['losses'][max(self.state['past_iters'] - self.n_iters_check_loss, 0):self.state['past_iters']])
        window_mean = window_sum / min(self.state['past_iters'], self.n_iters_check_loss)

        return 'Iter[{:d}] |\tLoss {:1.4f}'.format(self.state['past_iters'], window_mean)


    def get_epoch_mean(self, n_batches):
        print('n_batches:', n_batches)
        print(self.records['losses'][- n_batches:])
        return np.sum(self.records['losses'][- n_batches:]) / n_batches


    def save_model(self, model):
        if torch.typename(model).find('DataParallel') != -1:
            model = model.module

        model_file = os.path.join(self.save_path, 'model_{:d}_{:d}.pth'.format(self.state['past_epochs'], self.state['past_iters']))

        checkpoint = dict()
        checkpoint['state'] = self.state
        checkpoint['model'] = model.state_dict()

        torch.save(checkpoint, model_file)

        print('\n=> model snapshot saved to', model_file, '<=')


    def save_records(self):
        with open(self.record_path, 'w') as file:
            file.write(json.dumps(self.records))

        print('=> records saved to', self.record_path, '<=\n')


    def check_model(self, trainer):
        if self.state['past_iters'] % self.n_iters_check_model != 0:
            return

        self.records['metrics'][self.state['past_iters']] = trainer.eval(self.test_loader, torch.device('cuda'))

        self.save_model(trainer.get_model())
        self.save_records()
