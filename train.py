import os
import sys
import time
import argparse

import ray
from problems import get_problem
from strategies import get_strategy

from utils import CSVLogger


def run():
    assert params['env_name'] in ['HalfCheetah-v2', 'Ant-v2', 'Swimmer-v2', 'Hopper-v2', 'Walker2d-v2', 'Humanoid-v2']
    assert params['stg_name'] in ['es', 'ges', 'asebo', 'pes', 'pges']
    assert params['optim'] in ['bgd', 'sgd', 'adam']
    assert params['policy'] in ['linear', 'toeplitz']
    assert params['init_weight'] in ['zero', 'uniform']
    assert params['obs_norm'] in ['meanstd', 'no']
    assert params['fit_norm'] in ['div_std', 'z_score', 'rank', 'no']
    assert not params['pop_size'] & 1

    problem = get_problem("gym", params)
    solver = get_strategy(problem.params)
    solver.initialize()

    elapsed_time = 0.0
    start_time = time.time()
    iteration = 0
    while problem.total_steps <= params['max_steps']:

        if iteration % params['log_every'] == 0:
            elapsed_time += time.time() - start_time

            eval_info = problem.evaluate_rollouts(solver.mu)
            print("Time: {} | Iteration: {} | Total_steps: {} | Reward_mean: {}".format(elapsed_time, iteration, problem.total_steps, eval_info['mean']))
            sys.stdout.flush()

            logger.writerow({
                'time' : elapsed_time,
                'iteration' : iteration,
                'total_steps' : problem.total_steps,
                'reward_mean' : eval_info['mean'],
                'reward_std' : eval_info['std'],
                'reward_max' : eval_info['max'],
                'reward_min' : eval_info['min'],
                'alpha' : solver.params['alpha'],
                'sub_dims' : solver.params['sub_dims'],
                'pop_size' : solver.params['pop_size'],
            })

        X = solver.ask()
        Y, done = problem.aggregate_rollouts(X)
        solver.tell(-Y, done)
        
        iteration += 1
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--K', type=int, default=1000)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=20000000)

    parser.add_argument('--stg_name', type=str, default='asebo')
    parser.add_argument('--pop_size', type=int, default=400)
    parser.add_argument('--lrate', type=float, default=0.02)
    parser.add_argument('--sigma', type=float, default=0.02)

    # Guiding Subspace
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--sub_dims', type=int, default=1)

    # Asebo
    parser.add_argument('--warm_up', type=int, default=70)
    parser.add_argument('--threshold', type=float, default=0.995)
    parser.add_argument('--min', type=int, default=10)
    parser.add_argument('--decay', type=float, default=0.995)

    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--h_dim', type=int, default=32)
    parser.add_argument('--init_weight', type=str, default='zero')
    parser.add_argument('--obs_norm', type=str, default='no')
    parser.add_argument('--fit_norm', type=str, default='z_score')

    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./log')
    args = parser.parse_args()
    params = vars(args)

    expr_name = "env_{}-stg_{}-lr_{}-sig_{}-pop_{}-T_{}-K_{}-al_{}-sub_{}-opt_{}-pol_{}-init_{}-obs_{}-fit_{}".format(
            args.env_name,
            args.stg_name,
            args.lrate,
            args.sigma,
            args.pop_size,
            args.T,
            args.K,
            args.alpha,
            args.sub_dims,
            args.optim,
            args.policy,
            args.init_weight,
            args.obs_norm,
            args.fit_norm,
    )

    save_dir = os.path.join(args.save_dir, expr_name, 'seed_{}'.format(args.seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = CSVLogger(
        fieldnames = [
            'time', 'iteration', 'total_steps', 'reward_mean', 'reward_std', 
            'reward_max', 'reward_min', 'alpha', 'sub_dims', 'pop_size',
        ],
        filename = os.path.join(save_dir, 'record.csv')
    )

    ray.init()
    run()
