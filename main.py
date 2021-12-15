import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from train import Trainer
from writer import Writer
from options import args
from datasets import get_loader

import segmentation

def create_model(args):
    model = segmentation.assemble_and_load(args)
    state = None

    if args.resume or args.test_only:
        save_path = os.path.join(args.save_path, args.head + '_' + args.backbone + '_' + args.suffix)

        print('=> => loads checkpoint', os.path.join(save_path, args.model_name))

        fetch_dict = torch.load(os.path.join(save_path, args.model_name))['model']
        state_dict = model.state_dict()

        fetch_keys = set(fetch_dict.keys())
        state_keys = set(state_dict.keys())

        for key in fetch_keys.difference(state_keys):
            print('=> => => fetch key [', key, '] deleted due to redundancy')
            fetch_dict.pop(key)

        for state_key in state_keys.difference(set(fetch_dict.keys())):
            print('=> => => state key [', state_key, '] untended')

        state_dict.update(fetch_dict)
        model.load_state_dict(state_dict)
        print('<= <= checkpoint load is done')

    cudnn.benchmark = True
    model = nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda() if args.n_cudas != 1 else model.cuda()

    return model, state


def main():
    print('\n=> prepares a model')
    model, state = create_model(args)
    print('<= a model is ready')

    print('\n=> prepares a writer')
    writer = Writer(args, state)
    print('<= a writer is ready')

    print('\n=> prepares a trainer')
    trainer = Trainer(args, model, writer)
    print('<= a trainer is ready')

    if args.test_only:
        print('\n=> prepares data loaders')
        test_loader = get_loader(args, 'test')
        print('<= data loaders are ready')

        print('\n=> sets loaders')
        trainer.set_test_loader(test_loader)
        print('<= loaders are set')

        print('\n=> validation starts')
        writer.save_metrics(trainer.eval(torch.device('cuda'), args.save_spec))
        print('<= validation finishes')
    else:
        print('\n=> prepares data loaders')
        data_loader = get_loader(args, 'train')
        test_loader = get_loader(args, 'validation')
        print('<= data loaders are ready')

        print('\n=> sets loaders')
        trainer.set_loader(args, data_loader)
        trainer.set_test_loader(test_loader)
        print('<= loaders are set')

        print('\n=> train starts')
        while writer.next_epoch():
            trainer.train(torch.device('cuda'))
        print('<= train finishes')


if __name__ == '__main__':
    main()
