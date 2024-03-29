#%%
import ipdb as pdb

import time
import os
import argparse
import numpy as np
import random
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def fix_seed(seed):
  #Numpy
  np.random.seed(seed)
  #Random
  random.seed(seed)
  #Tensorflow
  tf.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="pursuit_and_evation", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/test/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=500, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./tmp/alpha1_2/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args(["--num-episodes", "10000", "--adv-policy", "maddpg", "--exp-name", "alpha0_2_0418_30000"])

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)              #出力層　（合計で三層）
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()     #create instance
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        #env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)        #done_callback=scenario.doneを追加(10/19)
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)  #episode style
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    #pdb.set_trace()
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,  #env.action_space = [Discrete(5),Discrete(5),Discrete(5),Discrete(5)]   obs_shape=[(16,),(16,),(16,),(14,)
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]    #各エージェントの観測値の次元を表す変数 #observation_space 観測したshapeの連続値:16 Box(16,).shape=(16,) evasion has (14,) 
        num_adversaries = min(env.n, arglist.num_adversaries)                   #敵対者の数（全エージェントと敵対者との小さいほう）自分でarglistのパラメーターを設定しないといけない
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)     #学習trainer
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))     #print policy

        # Initialize
        U.initialize()      #初期化？

        # Load previous results, if necessary
        if arglist.load_dir == "":               #locate save point
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()        #saverを用意している（sarver:学習後のパラメーターを使うため）
        obs_n = env.reset()             #最初のobs_nが与えられる
        episode_step = 0
        train_step = 0
        t_start = time.time()
        collide_list = []   
        save_collision = []         #collision回数を記録
        agent_pos = [[] for _ in range(len(env.world.agents))]      #agentのpositionを記録(10/12)
        step_len = []            #record episode len(10/19)


        print('Starting iterations...')
        while True:
            # get action    任意のエージェントの観測値を渡してactionを受け取る
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]            #何がactionとして与えられているのか？16つの要素を持った一次元配列
            #pdb.set_trace()
            #action_n = [np.array([0.01,0.01,0.01,0.96,0.01]), np.array([0.01,0.01,0.01,0.96,0.01])]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            #pdb.set_trace()
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                TD_error = agent.TDerror(trainers, obs_n, action_n, rew_n[i], new_obs_n)
                #TD_error = 1
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], abs(TD_error))    #Dに格納
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):     #報酬値の格納
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                #record step len (10/19)
                step_len.append(episode_step)
                episode_step = 0
                episode_rewards.append(0)
                #show number of collision
                for agent in env.world.agents:
                    if agent.adversary == False:
                        collide_list.append(agent.collide_num)
                        agent.collide_num = 0
                #end
                #show motion(10/19)
                if flag:
                    for i, agent in  enumerate(env.world.agents):
                        agent_pos[i].append(list(agent.state.p_pos))
                    print(agent_pos)
                    print("collision num:{}".format(env.world.agents[2].collide_num))
                    for i in range(len(env.world.agents)):
                        print("agent reward:{}".format(agent_rewards[i][-1]))
                    for landmark in env.world.landmarks:
                        print("landmark position:{}".format(landmark.state.p_pos))
                    flag = False
                    agent_pos = [[] for _ in range(len(env.world.agents))]
                #show motion end
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # to display, get position of object
            flag = False
            if (len(episode_rewards) == 10000) or (len(episode_rewards) == 20000) or (len(episode_rewards) == 15000) or (len(episode_rewards) == 5000):
                flag = True
                env.world.check = True
            if flag:
                for i, agent in  enumerate(env.world.agents):
                    agent_pos[i].append(list(agent.state.p_pos))
                if episode_step == 99:
                    print(agent_pos)
                    print("collision num:{}".format(env.world.agents[2].collide_num))
                    for i in range(len(env.world.agents)):
                        print("agent reward:{}".format(agent_rewards[i][-1]))
                    for landmark in env.world.landmarks:
                        print("landmark position:{}".format(landmark.state.p_pos))
                    flag = False
                    env.world.check = False
                    agent_pos = [[] for _ in range(len(env.world.agents))]

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()           #replay_sample_indexをNoneにするだけ
            for agent in trainers:
                loss = agent.update(trainers, train_step, len(episode_rewards))       #give one agent and step

            # save model, display training output
            if (terminal or done) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    print("the number of collision: {}".format(np.mean(collide_list[-arglist.save_rate:])))     #show number of collision(10/11)
                    save_collision.append(np.mean(collide_list[-arglist.save_rate:]))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3 )))
                    save_collision.append(np.mean(collide_list[-arglist.save_rate:]))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                print("the number of collision:{}".format(save_collision))
                #print(step_len)          #step数のヒストグラムを表示
                with open('rewards_alpha0_2_0418.txt', 'w') as f:
                    f.writelines(str(episode_rewards))
                break

if __name__ == '__main__':
    os.environ['PYTHONHASHSEED'] = '0'
    SEED = 42
    fix_seed(SEED)
    arglist = parse_args()
    train(arglist)

# %%
