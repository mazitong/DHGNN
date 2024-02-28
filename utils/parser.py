'''
Created
@author: Zitong Ma
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DHGNN.")
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')

    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Choose a dataset from {cora,survey,citeseer}')
    parser.add_argument('--split',nargs='?', default='/split0',
                        help='Split dataset')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[128,128]',
                        help='Output sizes of every layer')

    parser.add_argument('--regs', nargs='?', default='[1e-3]',
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay.')

    parser.add_argument('--milestones', nargs='?', default='[100, 500]',
                        help='milestones.')

    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Gamma.')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')


    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')


    return parser.parse_args()
