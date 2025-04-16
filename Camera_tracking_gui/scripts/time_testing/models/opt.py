from pprint import pprint
import argparse

def parse_args(er_override=None):
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='Semantic_Segmentation_Dataset/', help='name of dataset')
    # Optimization: General
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=250)
    parser.add_argument('--workers', type=int, help='Number of workers', default=4)
    parser.add_argument('--model', help='model name', default='densenet')
    parser.add_argument('--evalsplit', help='eval split', default='val')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save', help='save folder name', default='0try')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load', type=str, default=None, help='load checkpoint file name')
    parser.add_argument('--resume', action='store_true', help='resume train from load chkpoint')
    parser.add_argument('--test', action='store_true', help='test only')
    parser.add_argument('--savemodel', action='store_true', help='checkpoint save the model')
    parser.add_argument('--testrun', action='store_true', help='test run with few dataset')
    parser.add_argument('--expname', type=str, default='info', help='extra explanation of the method')
    parser.add_argument('--useGPU', type=str, default=True, help='Set it as False if GPU is unavailable')
    parser.add_argument('--device', type=str, default=None, help="Device to use (e.g., 'cuda', 'cuda:0', or 'cpu')")
    parser.add_argument('--er', type=int, default=6, help="Expansion Ratio value for Inverted Residual Blocks")
    parser.add_argument('--testsavedir', type=str, default="/home/hice1/snachimuthu7/scratch/test", help="Specified path to save test results in")

    # Parse arguments
    if er_override is not None:
        # If er_override is provided, use command-line style arguments
        args = parser.parse_args([f'--er={er_override}'])
    else:
        # Otherwise parse from command line
        args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    opt = parse_args()
    print('opt[\'dataset\'] is ', opt.dataset) 