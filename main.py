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

    model_name = args.head + '_' + args.backbone

    assert hasattr(segmentation, model_name)
    model = getattr(segmentation, model_name)(args.pretrain, args.stride, args.n_classes, args.aux_loss, args.stride_lift)
    state = None

    if args.resume or args.test_only:
        save_path = os.path.join(args.save_path, args.head + '_' + args.backbone + '_' + args.suffix)

        print('=> Loading checkpoint from ' + os.path.join(save_path, args.model_name))
        checkpoint = torch.load(os.path.join(save_path, args.model_name))

        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

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
