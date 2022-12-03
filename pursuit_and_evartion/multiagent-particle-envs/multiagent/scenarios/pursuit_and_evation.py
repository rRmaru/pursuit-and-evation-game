import ipdb as pdb

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1     #evation agent
        num_adversaries = 3     #pursuit agent
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True        #collider
            agent.silent = True         #cannot communication
            agent.adversary = True if i < num_adversaries else False    #if i is more than 3, agent.adversary is False
            
            agent.size = 0.075 if agent.adversary else 0.05     #adversary is bigger than good_agent
            agent.accel = 3.0 if agent.adversary else 4.0       #adversary is slower than good_agent
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3   #max_speed 
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            #agent.state.p_pos = np.array([0+i*0.5,-0.8])
            agent.state.p_pos = np.array([np.random.uniform(-1,0), np.random.uniform(-1,1)] if agent.adversary else [np.random.uniform(0,1), np.random.uniform(-1,1)])
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)      #positionを-1~1の範囲で設定
            agent.state.p_vel = np.zeros(world.dim_p)           #速度を二次元で0に設定
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)       #landmarkの位置を初期化
                #landmark.state.p_pos = np.array([0, 0.5])
                landmark.state.p_vel = np.zeros(world.dim_p)        #速度をゼロで設定


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)　#adversaryとの距離で報酬をもらうことができる
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 100
                    #collide num
                    agent.collide_num += 1

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        """def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)"""

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)  #good_agentとの距離によって報酬を受け取れるように設定できる
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 100
        return rew

    def observation(self, agent, world):            #ちょっとわからない
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)       #landmarkの位置からagentの位置を引く
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue     #もし今指定されているエージェントがotherと同じならcontinueで次に行く
            comm.append(other.state.c)      #ほかのエージェントのコミュニケーション内容を格納
            other_pos.append(other.state.p_pos - agent.state.p_pos)         #ほかのエージェントの場所を格納
            if not other.adversary:                         #敵ならばベクトルは取得しない！？ここでいう敵とは追跡者の事、逃亡者のベクトルだけ取得
                other_vel.append(other.state.p_vel)         #ほかのエージェントのベクトルを格納（方向のこと？）
        #pdb.set_trace()
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)       #すべてのリストをつないで返り値としている  1*16次元の配列

    #make sign to end episode when collision with agent and agent
    def done(self, agent, world):
        agents = self.good_agents(world)       #good_agent(逃亡者)がagents
        adversaries = self.adversaries(world)   #追跡者がadversaries
        if agent.collide:   #逃亡者が衝突判定を持っていると
            for ag in agents:       
                for adv in adversaries:     #誰かが衝突していれば
                    if self.is_collision(ag, adv):
                        return True         #when pursuit and evasion are collided, return True
        return False
