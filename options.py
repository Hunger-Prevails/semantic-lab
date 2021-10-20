import argparse

parser = argparse.ArgumentParser(description='parser for all pipeline configurations')

# model init options
model_init = parser.add_mutually_exclusive_group(True)
model_init.add_argument('-pretrain', action='store_true', help='whether to load an imagenet or coco pre-train')
model_init.add_argument('-resume', action='store_true', help='whether to continue from a previous checkpoint')

# bool options
parser.add_argument('-shuffle', action='store_true', help='shuffle train samples at the start of each epoch')
parser.add_argument('-half_acc', action='store_true', help='whether to use half precision for speed-up and memory efficiency')
parser.add_argument('-save_record', action='store_true', help='path to save train record')
parser.add_argument('-test_only', action='store_true', help='only performs test')
parser.add_argument('-colour', action='store_true', help='whether to perform colour augmentation')

# required options
parser.add_argument('-backbone', required=True, help='backbone architecture')
parser.add_argument('-head', required=True, help='head struction')
parser.add_argument('-model_path', help='path to an imagenet pre-train or checkpoint')
parser.add_argument('-suffix', required=True, help='Model suffix')
parser.add_argument('-data_name', required=True, help='name of dataset')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-criterion', required=True, help='criterion function for estimation loss')

# integer options
parser.add_argument('-n_epochs', default=20, type=int, help='number of total epochs')
parser.add_argument('-n_warmups', default=1, type=int, help='number of warmup iterations')
parser.add_argument('-n_cudas', default=2, type=int, help='number of cuda devices available')
parser.add_argument('-n_classes', default=19, type=int, help='number of joints in the dataset')
parser.add_argument('-n_workers', default=2, type=int, help='number of subprocesses to load data')
parser.add_argument('-batch_size', default=64, type=int, help='size of mini-batches for each iteration')
parser.add_argument('-queue_size', default=16, type=int, help='')
parser.add_argument('-stride', default=16, type=int, help='stride of network for train')

# learn rate options
parser.add_argument('-learn_rate', default=1e-3, type=float, help='base learn rate')
parser.add_argument('-learn_rate_decay', default=0.2, type=float, help='learn rate decay factor')
parser.add_argument('-learn_rate_warmup', default=0.1, type=float, help='learn rate decay for warmup phase')

# optimizer options
parser.add_argument('-grad_clip_norm', default=5.0, type=float, help='norm for gradient clip')
parser.add_argument('-grad_scale_factor', default=32.0, type=float, help='magnitude of loss scaling when computations are performed in half precision')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for gradient accumulation over the iterations')
parser.add_argument('-weight_decay', default=4e-5, type=float, help='weight decay factor for regularization')

args = parser.parse_args()
