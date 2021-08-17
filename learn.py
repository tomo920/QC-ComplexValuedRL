import numpy as np
import os

def save_episode(save_dir, result, env, agent):
    np.save('{0}/seed_state_{1}.npy'.format(save_dir, len(result)), np.random.get_state())
    np.save('{0}/env_info_{1}.npy'.format(save_dir, len(result)), env.get_info())
    np.save('{0}/agent_info_{1}.npy'.format(save_dir, len(result)), agent.get_info())

def load_episode(save_dir, result, env, agent):
    seed_state = np.load('{0}/seed_state_{1}.npy'.format(save_dir, len(result)), allow_pickle=True)
    np.random.set_state(tuple(seed_state))
    env_info = np.load('{0}/env_info_{1}.npy'.format(save_dir, len(result)), allow_pickle=True)
    env.load_info(env_info)
    observation = env.get_observation()
    agent_info = np.load('{0}/agent_info_{1}.npy'.format(save_dir, len(result)), allow_pickle=True)
    agent.load_info(agent_info)
    return observation

def learning(config,
             env,
             agent,
             seed,
             save_dir):
    '''
    Execute one epoch of learning
    or Execute test using learned parameter.
    '''
    result = []
    # start learning or test
    for i in range(config.n_episodes):
        if config.is_restore and len(result) == 0 and os.path.exists('{0}/test_result_{1}.npy'.format(save_dir, config.test_save_name)):
            result = np.load('{0}/test_result_{1}.npy'.format(save_dir, config.test_save_name), allow_pickle=True)
            result = list(result)
            observation = load_episode(save_dir, result, env, agent)
        elif config.is_restore and len(result) == 0 and os.path.exists('{0}/seed_state_{1}.npy'.format(save_dir, len(result))):
            # restart in 0 episode
            observation = load_episode(save_dir, result, env, agent)
        else:
            observation = env.reset()
            agent.init_history(observation)
            if config.is_restore:
                save_episode(save_dir, result, env, agent)
        while True:
            pi = agent.get_policy(observation, i)
            if config.is_legal_action:
                action = np.random.choice(env.legal_action_list[observation], p = pi)
            else:
                action = np.random.choice(env.action_size, p = pi)
            next_observation, reward, done = env.step(action)
            agent.update(observation, action, next_observation, reward, done)
            if done:
                result.append(env.steps)
                if config.mode == 'learn':
                    np.save('{0}/result_{1}.npy'.format(save_dir, seed), result)
                elif config.mode == 'test':
                    np.save('{0}/test_result_{1}.npy'.format(save_dir, config.test_save_name), result)
                    if config.save_log:
                        np.save('{0}/log_data_{1}.npy'.format(save_dir, len(result)), agent.get_log())
                break
            else:
                observation = next_observation
                if config.is_restore:
                    save_episode(save_dir, result, env, agent)
        if config.is_restore and len(result) == config.n_episodes:
            break
        if config.mode == 'learn' and i % config.save_freq == config.save_freq - 1:
            np.save('{0}/params_{1}.npy'.format(save_dir, i), agent.get_params())
    if config.mode == 'learn':
        np.save('{}/params_last.npy'.format(save_dir), agent.get_params())
