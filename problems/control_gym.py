import ray
import gym
import numpy as np
from utils import get_policy


@ray.remote
class Worker(object):

    def __init__(self, params):

        self.policy = get_policy(params)

    def do_rollout(self, vec, env_pack, roll_len):

        self.policy.set_weight(vec)

        env = env_pack[0]
        env.sim.set_state(env_pack[1])

        total_reward = 0
        obs = env_pack[2]
        for steps in range(1, roll_len + 1):
            action = self.policy.evaluate(obs)
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            action = action.reshape(len(action), )
            obs, rew, done, _ = env.step(action)
            total_reward += rew

            if done:
                break

        return total_reward, done, steps, (env, env.sim.get_state(), obs)


class Master(object):

    def __init__(self, params):

        self.params = params
        self.rng = np.random.RandomState(params['seed'])

        self.env_buffer = [()] * self.params['pop_size']
        self.reset(np.arange(self.params['pop_size']))

        _env = gym.make(self.params['env_name'])
        self.params['ob_dim'] = _env.observation_space.shape[0]
        self.params['ac_dim'] = _env.action_space.shape[0]
        self.workers = [Worker.remote(self.params)] * self.params['num_worker']
        
        self.total_steps = 0

    def reset(self, idxs):

        new_seeds = self.rng.randint(0, 100000, len(idxs))

        for i in range(len(idxs)):
            env = gym.make(self.params['env_name'])
            env._max_episode_steps = self.params['T']
            env.seed(int(new_seeds[i]))
            obs = env.reset()
            self.env_buffer[idxs[i]] = (env, env.sim.get_state(), obs)

    def aggregate_rollouts(self, A):

        rollout_ids, worker_id = [], 0
        for i in range(self.params['pop_size']):
            rollout_ids += [self.workers[worker_id].do_rollout.remote(A[i, :], self.env_buffer[i], self.params['K'])]
            worker_id    = (worker_id + 1) % self.params['num_worker']
        results = ray.get(rollout_ids)
        
        total_rewards, is_done = [], []
        for i, res in enumerate(results):
            total_rewards.append(res[0])
            is_done.append(res[1])
            self.total_steps  += res[2]
            self.env_buffer[i] = res[3]

        for i in range(self.params['pop_size']):
            if is_done[i]:
                is_done[self.params['pop_size'] - 1 - i] = True

        if any(is_done):
            self.reset(np.where(is_done)[0])

        return np.array(total_rewards)

    def evaluate_rollouts(self, vec, num_evals=10):

        eval_seeds = self.rng.randint(0, 100000, num_evals)
        
        rollout_ids, worker_id = [], 0
        for i in range(num_evals):
            env = gym.make(self.params['env_name'])
            env._max_episode_steps = self.params['T']
            env.seed(int(eval_seeds[i]))
            obs = env.reset()
            env_pack = (env, env.sim.get_state(), obs)
            rollout_ids += [self.workers[worker_id].do_rollout.remote(vec, env_pack, self.params['T'])]
            worker_id    = (worker_id + 1) % self.params['num_worker']
        results = ray.get(rollout_ids)

        eval_rewards = []
        for res in results:
            eval_rewards.append(res[0])

        return {
                    "mean" : np.mean(eval_rewards),
                    "std"  : np.std(eval_rewards),
                    "max"  : np.max(eval_rewards),
                    "min"  : np.min(eval_rewards),
                }
