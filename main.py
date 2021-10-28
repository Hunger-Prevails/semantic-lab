import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from options import args
from datasets import get_loader
from writer import Writer
from train import Trainer

import segmentation

def create_model(args):

    model_name = args.head + '_' + args.backbone

    assert hasattr(segmentation, model_name)
    model = getattr(segmentation, model_name)(args.pretrain, True, args.n_classes, args.aux_loss)
    state = None

    if args.resume or args.test_only:
        print('=> Loading checkpoint from ' + args.model_path)
        checkpoint = torch.load(args.model_path)

        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    cudnn.benchmark = True
    model = nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda() if args.n_cudas != 1 else model.cuda()

    return model, state


def main():
    print('\n=> prepares a model')
    model, state = create_model(args)
    print('<= a model is ready')

    print('\n=> prepares data loaders')
    if args.test_only:
        test_loader = get_loader(args, 'test')
    else:
        test_loader = get_loader(args, 'validation')
        data_loader = get_loader(args, 'train')
    print('<= data loaders are ready')

    print('\n=> prepares a writer')
    writer = Writer(args, state, test_loader)
    print('<= a writer is ready')

    print('\n=> prepares a trainer')
    trainer = Trainer(args, model, writer)
    print('<= a trainer is ready')

    if args.test_only:
        print('\n=> validation starts')
        test_rec = trainer.eval(test_loader, torch.device('cuda'))
        print('<= validation finishes')
    else:
        print('\n=> train starts')
        start_epoch = writer.state['past_epochs'] + 1

        for epoch in range(start_epoch, args.n_epochs + 1):
            trainer.train(epoch, data_loader, torch.device('cuda'))

        print('<= train finishes')

if __name__ == '__main__':
    main()
