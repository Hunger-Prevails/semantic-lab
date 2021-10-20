import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import args
from datasets import get_data_loader
from writer import Writer
from train import Trainer

from resnet import resnet18
from resnet import resnet50


def get_catalogue():
    model_creators = dict()

    model_creators['resnet18'] = resnet18
    model_creators['resnet50'] = resnet50

    return model_creators


def create_model(args):

    assert not (args.resume and args.pretrain)

    state = None

    model_creators = get_catalogue()

    assert args.model in model_creators

    model = model_creators[args.model](args)

    if args.test_only or args.val_only:
        save_path = os.path.join(args.save_path, args.model + '-' + args.suffix)

        print('=> Loading checkpoint from ' + os.path.join(save_path, 'best.pth'))
        assert os.path.exists(save_path)

        best = torch.load(os.path.join(save_path, 'best.pth'))
        best = best['best'];
        
        checkpoint = os.path.join(save_path, 'model_%d.pth' % best)
        checkpoint = torch.load(checkpoint)['model']

        keys = checkpoint.keys()
        model_dict = model.state_dict()

        for key in keys:
            if key not in model_dict:
                print key
                del checkpoint[key]
        
        model.load_state_dict(checkpoint)

    if args.resume:
        print('=> Loading checkpoint from ' + args.model_path)
        checkpoint = torch.load(args.model_path)
        
        model.load_state_dict(checkpoint['model'])
        state = checkpoint['state']

    cudnn.benchmark = True
    model = model.cuda() if args.n_cudas == 1 else nn.DataParallel(model, device_ids = range(args.n_cudas)).cuda()

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
            trainer.train(epoch, data_loader)

        writer.final_print()

if __name__ == '__main__':
    main()
