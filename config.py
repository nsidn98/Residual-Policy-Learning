import argparse

parser = argparse.ArgumentParser(description='DDPG agent')
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                    help='the name of this experiment')
parser.add_argument('--gym-id', type=str, default="FetchPickAndPlace-v1",
                    help='the id of the gym environment')

# algorithm parameters
parser.add_argument('--actor_lr', type=float, default=1e-3,
                    help='the learning rate of the actor optimizer')
parser.add_argument('--critic_lr', type=float, default=1e-3,
                    help='the learning rate of the critic optimizer')
parser.add_argument('--polyak', type=float, default=0.95,
                    help='Polyak averaging coefficient')
parser.add_argument('hidden_dims', type=list, default=[256,256,256],
                    help='Hidden layer dimensions')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'],
                    help='Activation function for networks')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch Size for training')
parser.add_argument('--epsilon', type=float, default=0.3,
                    help='Probability of taking random actions')
parser.add_argument('--buffer_size', type=int, default=int(1e6),
                        help='the replay memory buffer size')
parser.add_argument('--total_timesteps', type=int, default=int(6e6),
                    help='total timesteps of the experiments')


parser.add_argument('--seed', type=int, default=0,
                    help='seed of the experiment')
parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                    help='if toggled, `torch.backends.cudnn.deterministic=False`')


# Algorithm specific arguments
parser.add_argument('--gamma', type=float, default=0.99,
                    help='the discount factor gamma')
parser.add_argument('--tau', type=float, default=0.005,
                    help="target smoothing coefficient (default: 0.005)")
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='the maximum norm for the gradient clipping')
parser.add_argument('--batch-size', type=int, default=256,
                    help="the batch size of sample from the reply memory")
parser.add_argument('--exploration-noise', type=float, default=0.1,
                    help='the scale of exploration noise')
parser.add_argument('--learning-starts', type=int, default=25e3,
                    help="timestep to start learning")
parser.add_argument('--policy-frequency', type=int, default=2,
                    help="the frequency of training policy (delayed)")
parser.add_argument('--noise-clip', type=float, default=0.5,
                        help='noise clip parameter of the Target Policy Smoothing Regularization')
args = parser.parse_args()