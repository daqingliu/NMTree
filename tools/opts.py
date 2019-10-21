import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split_by', type=str, required=True)
    parser.add_argument('--grounding_model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--glove', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--use_bn', default=False, action='store_true')
    parser.add_argument('--use_word_attn', default=False, action='store_true')

    # File path settings
    parser.add_argument('--checkpoint_path', type=str, default='log')
    parser.add_argument('--feats_path', type=str, default='data/feats')
    parser.add_argument('--data_file', default='data_plain')
    parser.add_argument('--visual_feat_file', default='matt_res_gt_feats.pth')

    # Network settings
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--rnn_type', type=str, default='lstm')
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_dirs', type=int, default=2)
    parser.add_argument('--att_size', type=int, default=1024)
    parser.add_argument('--word_size', type=int, default=300)
    parser.add_argument('--tag_size', type=int, default=50)
    parser.add_argument('--dep_size', type=int, default=50)

    # Optimization: general
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0, help="-1 for dont")
    parser.add_argument('--learning_rate_decay_every', type=int, default=10)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.9)
    parser.add_argument('--val_use', type=int, default=-1)
    parser.add_argument('--save_checkpoint_every', type=int, default=1)
    parser.add_argument('--losses_log_every', type=int, default=10)

    # Optimization
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--optim_alpha', type=float, default=0.8)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=666, help='random number generator seed to use')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
