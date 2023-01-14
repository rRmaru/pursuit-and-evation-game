import ipdb as pdb

import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):     #これが謎
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    #pdb.set_trace()
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):    #actor
    with tf.variable_scope(scope, reuse=reuse):         #name space
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  #[SoftCategoricalPdType(5)*4]　return 

        # set up placeholders
        obs_ph_n = make_obs_ph_n        #placeholder, BatchInput()
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]      #placeholder

        p_input = obs_ph_n[p_index] #p_index = i BatchInput()shapeがそれぞれで違う
        #pdb.set_trace()

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)       #input mlp  param_shape = 5　入力Placeholder 出力5
        #pdb.set_trace()
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)   #return SoftCategoricalPd(p)

        act_sample = act_pd.sample()                    #softmax関数を返す
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))  #政策の勾配方向

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]       #Q値の勾配方向
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])       #loss(損失関数)を計算
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)            #この部分がactを与える部分　観測値=input act_sample = softmax関数
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)         #action算出と一緒
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        #pdb.set_trace()

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()         #softmax関数　　action算出と一緒nnを用いてる
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64): #critic
    with tf.variable_scope(scope, reuse=reuse):     #名前空間→trainer  再利用→None
        # create distribtuions分布
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]   #act_space [Discrete(5)*4]   return SoftCategoricalPdType(5)*4

        # set up placeholders
        obs_ph_n = make_obs_ph_n        #placeholder, BatchInput()
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]    #[Discrete(5)*4] tf.Placeholder(float32, [None, 5], action+i)
        target_ph = tf.placeholder(tf.float32, [None], name="target")  #None→任意のshapeを割り当てる

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)         #obs_ph_n and act_ph_n二個目の要素を結合　ほかのエージェントの観測値も使用している
        if local_q_func:        #DDPGの場合MADDPGでは使用しない
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        #pdb.set_trace()
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]        #input mlp  ほかのエージェントの情報も得ている out_put = 1?
        #pdb.set_trace()
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))     #absolute_scope_vars = 名前空間の絶対パスを得る scope_vars = ?

        q_loss = tf.reduce_mean(tf.square(q - target_ph))       #損失関数の設定　与えられたリストに入っている数値の平均値を求める関数 二乗平均誤差

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg       損失関数

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]      #q_input = obs_ph_n + act_ph_n  qと同じ
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)         #すべての観測値と行動のセットとtarget_qを引数として入力
        #pdb.set_trace()

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):       #obs_shape_n=[(16,),(16,),(16,),(14,)] act_space_n=[Discrete(5)*4]
        self.name = name
        self.n = len(obs_shape_n)       #the number of agent  obs_shape_n = [(16,),(16,),(16,),(14,)] therefor n = 4
        self.agent_index = agent_index  #index = i
        self.args = args                #実行時のコマンドライン引数
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())    #placeholderを渡してる obs_shape_n[i] = (16,) or (14,) tf.placeholder(dtype, [None,16], name=name)

        # Create all the functions necessary to train the model         わからん
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(          #ここもわからん
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)          #decide replay buffer big
        #self.TDerror_buffer = Memory_TDerror(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done)) #doneをfloat型としてバッファに格納している 

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):        #main train part
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps  100ごとにupdateする
            return

        #pdb.set_trace()
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size) 
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):                         #自身の情報だけ得る 4回
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)     #sample_index バッチサイズの数だけサンプルを入手
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)          #dont know(これいる？)→rewの部分で必要

        # train q network
        num_sample = 1      #sample数
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]     #バッチサイズ分だけ一気に代入している 1024こ分のactionが出てくる
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))   #次の行動と次の観測　バッチサイズ分学習 *はアンパック（リストの要素をばらばらにして渡す）
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next            #Q値の計算 出力は1つ　doneが位置の場合は更新しない
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))        #target_qを教師として損失関数を導出
        
        #pdb.set_trace()

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))     #actorの損失関数を導出

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
    
    def TDerror(self, agents, j, obs_n, act_n, rew_n, obs_next_n):
        #reshape
        obs_next_n = self.reshape(obs_next_n)
        obs_n = self.reshape(obs_n)
        act_n = self.reshape(act_n)
        #pdb.set_trace()
        target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q = rew_n[j] + self.args.gamma * target_q_next
        q_main = self.q_debug['q_values'](*(obs_n + act_n))
        
        TD_error = target_q - q_main
        return TD_error
    
    def reshape(self, list):
        temp = []
        for i in range(len(list)):
            temp.append(np.array([list[i]]))
        return temp