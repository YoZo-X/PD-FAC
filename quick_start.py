from __future__ import absolute_import
import os
import argparse
import torch
import numpy as np

from learner.c51_rnn import C51_RNN
from utils import MapInfo, Env
from utils.networks import C51RNNNet
from mixer.tdn import TDN_Mixer


parser = argparse.ArgumentParser(description='marl-searcher')
parser.add_argument('--map_path', default='maps/Museum_network.csv',
                    help='Map file path')
parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                    help="Directory Path to store results (default: %(default)s)")
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Enforces no cuda usage (default: %(default)s)')
parser.add_argument('--train', action='store_true', default=True,
                    help='Trains the model')
parser.add_argument('--n_inter', default=51,
                    help='The number of intervals.')
parser.add_argument('--n_agents', default=3,
                    help='The number of agents.')
parser.add_argument('--v_max', default=5,
                    help='V max')
parser.add_argument('--v_min', default=-35,
                    help='V min')
parser.add_argument('--test', action='store_true', default=False,
                    help='Evaluates the model')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate (default: %(default)s)')
parser.add_argument('--discount', type=float, default=0.95,
                    help=' Discount rate (or Gamma) for TD error (default: %(default)s)')
parser.add_argument('--train_episodes', type=int, default=10000,
                    help='Learning rate (default: %(default)s)')
parser.add_argument('--test_episodes', type=int, default=100,
                    help='test episodes (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Learning rate (default: %(default)s)')
parser.add_argument('--mem_len', type=int, default=100000,
                    help='Learning rate (default: %(default)s)')
parser.add_argument('--seed', type=int, default=0,
                    help='seed (default: %(default)s)')
parser.add_argument('--test_interval', type=int, default=100,
                    help='Test interval (default: %(default)s)')
parser.add_argument('--force', action='store_true', default=True,
                    help='Trains the model')
args = parser.parse_args()


def main():
    device = 'cuda' if ((not args.no_cuda) and torch.cuda.is_available()) else 'cpu'
    args.env_result_dir = os.path.join(args.result_dir, args.map_path.split('/')[-1].split('.')[0])
    _path = os.path.join(args.env_result_dir, 'TDN', 'runs')
    print('Using {}'.format(device))

    if args.train and os.path.exists(_path) and os.listdir(_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to override it'.format(_path))
        else:
            import shutil

            shutil.rmtree(_path)
            os.makedirs(_path)
    else:
        if not os.path.exists(_path):
            os.makedirs(_path)

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize environment
    map1 = MapInfo(args.map_path)
    env_fn = lambda: Env(map1, n_agents=args.n_agents, max_steps=500, full_observation=False, view_target=False,
                         view_intarget=True, view_step=False, is_onehot=True)
    env = env_fn()
    env.seed(args.seed)
    net_fn = lambda: C51RNNNet(env, args.seed, args.n_inter, args.v_min, args.v_max, device=device)
    mixer_fn = lambda: TDN_Mixer(env, 51, True)
    algo = C51_RNN(env_fn, net_fn, mixer_fn, lr=args.lr, discount=args.discount, batch_size=args.batch_size,
                device=device, mem_len=args.mem_len, tau=0.01, path=_path,
                train_episodes=args.train_episodes, episode_max_steps=500)

    if args.train:
        algo.train(test_interval=args.test_interval)
    if args.test:
        algo.restore()
        test_score = algo.test(episodes=args.test_episodes, render=True, log=False, record=False)
        print(test_score)


if __name__ == "__main__":
    main()
