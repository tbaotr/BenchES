import ray, gym
import numpy as np
from utils import get_policy, get_obs_norm, get_act_norm


@ray.remote
class Worker(object):

    def __init__(self, params):

        self.policy = get_policy(params)
        self.act_norm = get_act_norm(params)

    def do_rollout(self, vec, env_pack, roll_len, train=True):

        self.policy.set_weight(vec)

        env = env_pack[0]
        env.sim.set_state(env_pack[1])
        obs_norm = env_pack[2]

        total_reward = 0
        obs = env_pack[3]
        for steps in range(1, roll_len + 1):
            n_obs = obs_norm(obs, update=train)
            action = self.policy.evaluate(n_obs)
            action = self.act_norm(action, env.action_space.low[0], env.action_space.high[0])
            action = action.reshape(-1)
            obs, rew, done, _ = env.step(action)
            total_reward += rew

            if done:
                break

        return total_reward, done, steps, (env, env.sim.get_state(), obs_norm, obs)


class Master(object):

    def __init__(self, params):

        self.params = params
        self.rng = np.random.RandomState(params['seed'])

        _env = gym.make(params['env_name'])
        params['ob_dim'] = _env.observation_space.shape[0]
        params['ac_dim'] = _env.action_space.shape[0]
        self.workers = [Worker.remote(params) for _ in range(params['num_worker'])]
        
        self.obs_norm = get_obs_norm(params)

        self.env_buffer = [()] * params['pop_size']
        self.reset_buffer(np.arange(params['pop_size']))

        self.total_steps = 0

    def reset_buffer(self, idxs):

        new_seeds = self.rng.randint(0, 100000, len(idxs))

        for i in range(len(idxs)):
            env = gym.make(self.params['env_name'])
            env._max_episode_steps = self.params['T']
            env.seed(int(new_seeds[i]))
            obs_norm = self.obs_norm.copy()
            obs_norm.stats_increment()
            obs = env.reset()
            self.env_buffer[idxs[i]] = (env, env.sim.get_state(), obs_norm, obs)

    def aggregate_rollouts(self, A):

        if self.params['pop_size'] < len(self.env_buffer):
            
            t = (len(self.env_buffer)-self.params['pop_size'])//2
            idx_pos, idx_neg = [], []
            for i in range(t):
                idx_pos.append(self.idx_done[i])
                idx_neg.append(self.idx_done[i+len(self.idx_done)//2])

            for i in reversed(idx_pos+idx_neg):
                self.env_buffer.pop(i)

        elif self.params['pop_size'] > len(self.env_buffer):
            
            t = (self.params['pop_size']-len(self.env_buffer))//2
            self.env_buffer = self.env_buffer[:len(self.env_buffer)//2] \
                            + [()] * t \
                            + self.env_buffer[len(self.env_buffer)//2:] \
                            + [()] * t

            idx_reset = np.concatenate([
                    np.arange(len(self.env_buffer)//2-t, len(self.env_buffer)//2),
                    np.arange(len(self.env_buffer)-t, len(self.env_buffer)),
                ])
            self.reset_buffer(idx_reset)


        rollout_ids, worker_id = [], 0
        for i in range(self.params['pop_size']):
            rollout_ids += [self.workers[worker_id].do_rollout.remote(A[i, :], self.env_buffer[i], self.params['K'], train=True)]
            worker_id = (worker_id + 1) % self.params['num_worker']
        results = ray.get(rollout_ids)
        
        total_rewards, is_done = [], []
        for i, res in enumerate(results):
            total_rewards.append(res[0])
            is_done.append(res[1])
            self.total_steps  += res[2]
            self.env_buffer[i] = res[3]

        for env_pack in self.env_buffer:
            self.obs_norm.update(env_pack[2])
        self.obs_norm.stats_increment()
        self.obs_norm.clear_buffer()

        for i in range(self.params['pop_size']):
            if is_done[i]:
                is_done[self.params['pop_size'] - 1 - i] = True
        self.idx_done = np.where(is_done)[0]

        if any(is_done):
            self.reset_buffer(self.idx_done)

        return np.array(total_rewards), self.idx_done

    def evaluate_rollouts(self, vec, num_evals=10):

        eval_seeds = self.rng.randint(0, 100000, num_evals)
        
        rollout_ids, worker_id = [], 0
        for i in range(num_evals):
            env = gym.make(self.params['env_name'])
            env._max_episode_steps = self.params['T']
            env.seed(int(eval_seeds[i]))
            obs_norm = self.obs_norm.copy()
            obs_norm.stats_increment()
            obs = env.reset()
            env_pack = (env, env.sim.get_state(), obs_norm, obs)
            rollout_ids += [self.workers[worker_id].do_rollout.remote(vec, env_pack, self.params['T'], train=False)]
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
