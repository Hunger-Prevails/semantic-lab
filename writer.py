import os
import json
import torch
import numpy as np

class Writer:

    def __init__(self, args, state):
        self.n_epochs = args.n_epochs

        self.n_iters_check_loss = args.n_iters_check_loss
        self.n_iters_check_model = args.n_iters_check_model

        self.state = state if state else dict(past_epochs = 0, past_iters = 0)

        self.save_path = os.path.join(args.save_path, args.head + '_' + args.backbone + '_' + args.suffix)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        if not args.test_only:
            self.save_configs(args)

        self.record_path = os.path.join(self.save_path, 'records.json')
        if args.resume:
            assert os.path.exists(self.record_path)

            with open(self.record_path, 'r') as file:
                records = json.load(file)

            self.records = self.clamp_to_state(records)
            self.inc_epoch()
        else:
            self.records = dict(losses = list(), metrics = dict())


    def inc_iter(self, loss):
        self.records['losses'].append(loss)
        self.state['past_iters'] += 1


    def inc_epoch(self):
        self.state['past_epochs'] += 1


    def next_epoch(self):
        return self.state['past_epochs'] < self.n_epochs


    def current_epoch(self):
        return self.state['past_epochs'] + 1


    def clamp_to_state(self, records):
        records['losses'] = records['losses'][:self.state['past_iters']]
        records['metrics'] = dict([item for item in records['metrics'].items() if int(item[0]) <= self.state['past_iters']])
        return records


    def print_iter(self, i_batch, n_batches):
        window_sum = np.sum(self.records['losses'][max(self.state['past_iters'] - self.n_iters_check_loss, 0):self.state['past_iters']])
        window_mean = window_sum / min(self.state['past_iters'], self.n_iters_check_loss)

        message = 'train Epoch[{:d}] [{:d}:{:d}]'.format(self.current_epoch(), i_batch + 1, n_batches)

        print('=> => | {:s} | Loss = {:1.4f}'.format(message, window_mean))


    def print_epoch(self, n_batches):
        epoch_mean = np.sum(self.records['losses'][- n_batches:]) / n_batches

        print('=> | train Epoch[{:d}] finishes | Epoch-Mean: {:1.4f} <=\n'.format(self.current_epoch(), epoch_mean))


    def save_model(self, model):
        if torch.typename(model).find('DataParallel') != -1:
            model = model.module

        model_file = os.path.join(self.save_path, 'model_{:03d}_{:06d}.pth'.format(self.state['past_epochs'] + 1, self.state['past_iters']))

        checkpoint = dict()
        checkpoint['state'] = self.state
        checkpoint['model'] = model.state_dict()

        torch.save(checkpoint, model_file)

        print('\n=> model snapshot saved to', model_file, '<=')


    def save_records(self):
        with open(self.record_path, 'w') as file:
            file.write(json.dumps(self.records))

        print('=> records saved to', self.record_path, '<=\n')


    def save_configs(self, args):
        configs_path = os.path.join(self.save_path, 'configs.json')
        with open(configs_path, 'w') as file:
            file.write(json.dumps(vars(args)))


    def check_model(self, trainer):
        if self.state['past_iters'] % self.n_iters_check_model != 0:
            return

        self.records['metrics'][str(self.state['past_iters']).zfill(6)] = trainer.eval(torch.device('cuda'))

        self.save_model(trainer.get_model())
        self.save_records()


    def save_spec(self, predictions, i_batch):
        spec_path = os.path.join(self.save_path, 'predictions')

        if not os.path.exists(spec_path):
            os.mkdir(spec_path)

        spec_name = os.path.join(spec_path, 'batch-{:04d}'.format(i_batch))
        np.save(spec_name, predictions)


    def save_metrics(self, metrics):
        metrics_name = os.path.join(self.save_path, 'metrics.json')

        with open(metrics_name, 'w') as file:
            file.write(json.dumps(metrics))
