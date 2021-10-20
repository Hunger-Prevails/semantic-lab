import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import args
from datasets import get_data_loader
from writer import Writer
from train import Trainer

import segmentation

def create_model(args):

    model_name = args.head + '_' + args.backbone

    assert hasattr(segmentation, model_name)
    model = getattr(segmentation, model_name)(args.pretrain, True, args.n_classes)
    state = None

    if args.resume or args.test_only:
        print('=> Loading checkpoint from ' + args.model_path)
        checkpoint = torch.load(args.model_path)

        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    cudnn.benchmark = True
    nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda() if args.n_cudas != 1 else model = model.cuda()

    return model, state


def main():
    model, state = create_model(args)
    print('=> Model and criterion are ready')

    if args.test_only:
        test_loader = get_data_loader(args, 'test')
        print('=> Dataloaders are ready')
    else:
        test_loader = get_data_loader(args, 'valid')
        data_loader = get_data_loader(args, 'train')
        print('=> Dataloaders are ready')

    writer = Writer(args, state)
    print('=> Writer is ready')

    trainer = Trainer(args, model, writer)
    print('=> Trainer is ready')

    if args.test_only:
        test_rec = trainer.test(test_loader)

    else:
        start_epoch = writer.state['epoch_done'] + 1
        print('=> Train process starts')

        for epoch in xrange(start_epoch, args.n_epochs + 1):
            trainer.train(epoch, data_loader, torch.device('cuda'))

        writer.final_print()

if __name__ == '__main__':
    main()
