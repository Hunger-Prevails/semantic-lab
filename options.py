import argparse

parser = argparse.ArgumentParser(description='parser for all pipeline configurations')

# model init options
model_init = parser.add_mutually_exclusive_group(required = True)
model_init.add_argument('-pretrain', action='store_true', help='loads a coco pre-train if true otherwise loads a imagenet pre-train')
model_init.add_argument('-resume', action='store_true', help='whether to continue from a previous checkpoint')

# bool options
parser.add_argument('-shuffle', action='store_true', help='shuffle train samples at the start of each epoch')
parser.add_argument('-half_acc', action='store_true', help='whether to use half precision for speed-up and memory efficiency')
parser.add_argument('-test_only', action='store_true', help='only performs test')
parser.add_argument('-enc_flip', action='store_true', help='whether to perform random flip augmentation')
parser.add_argument('-enc_crop', action='store_true', help='whether to perform random crop augmentation')
parser.add_argument('-enc_colour', action='store_true', help='whether to perform random colour perturbation')
parser.add_argument('-aux_loss', action='store_true', help='whether to introduce an auxiliary loss term on intermediate feature maps')

# required options
parser.add_argument('-backbone', required=True, help='backbone architecture')
parser.add_argument('-head', required=True, help='head struction')
parser.add_argument('-model_path', help='path to a checkpoint model')
parser.add_argument('-suffix', required=True, help='Model suffix')
parser.add_argument('-data_name', required=True, help='name of dataset')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-criterion', required=True, help='criterion function for estimation loss')

# integer options
parser.add_argument('-n_epochs', default=100, type=int, help='number of total epochs')
parser.add_argument('-n_iters_start', default=256, type=int, help='number of iterations in the warmup phase')
parser.add_argument('-n_iters_check_loss', default=32, type=int, help='number of iterations over which to average the losses')
parser.add_argument('-n_iters_check_model', default=512, type=int, help='number of iterations before next validation checkpoint')
parser.add_argument('-n_cudas', default=2, type=int, help='number of cuda devices available')
parser.add_argument('-n_classes', default=36, type=int, help='number of joints in the dataset')
parser.add_argument('-n_workers', default=2, type=int, help='number of subprocesses to load data')
parser.add_argument('-batch_size', default=64, type=int, help='size of mini-batches for each iteration')
parser.add_argument('-stride', default=16, type=int, help='stride of network for train')

# learn rate options
parser.add_argument('-learn_rate', default=1e-2, type=float, help='base learn rate')
parser.add_argument('-learn_rate_start', default=0.1, type=float, help='start learn rate')
parser.add_argument('-learn_rate_decay', default=0.6, type=float, help='learn rate decay for polynomial schedule')
parser.add_argument('-learn_rate_bottom', default=1e-5, type=float, help='bottom learn rate')

# float options
parser.add_argument('-crop_rate', default=0.75, type=float, help='boundary crop rate for random crop')
parser.add_argument('-weight_decay', default=5e-4, type=float, help='weight decay factor for regularization')
parser.add_argument('-grad_clip_norm', default=5.0, type=float, help='norm for gradient clip')
parser.add_argument('-grad_scale_factor', default=32.0, type=float, help='loss scale multiplier for half precision backward pass')

args = parser.parse_args()
