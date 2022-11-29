import ipdb as pdb

import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):       #set when step
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # the number of collision
        self.collide_num = 0

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2  #次元数
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):     #どれぐらい移動するか決定する
        # set actions for scripted agents 
        for agent in self.scripted_agents:      #これは今回は関係ない？
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)       #agents and landmarks sum number  [None]のリストを作成
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # bound entitiy
        self.bound()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:           # add noise 
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0            #ここでaction.u（stepでせっていした）を用いる
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities  環境から受ける力
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response 全探索
        for a,entity_a in enumerate(self.entities):  #all entities(agents and landmarks)
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue    #同じ物体は含まない
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)         #衝突時の力を受ける
                if(f_a is not None):            #exit f_a
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):      #all entities (agents and landmarks)
            if not entity.movable: continue            # movable is False continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)        #damping = 0.25
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt      #initial_mas = 1.0
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))        #p_vel(ベクトル)を計算
                if speed > entity.max_speed:            #speed がmax_speedを超えていたら
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed      #ベクトルをベクトルの絶対値（speed）でわる（正規化？）×max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt          #ベクトル×時間 = 距離なのでagentの距離を算出できる

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):        #aかbどちらかでもcoollideがFalse
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself  同じ物体の時
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin             #contact_margin = 0.001
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k       #基本的に負だが、接触すると急激に大きくなる
        #pdb.set_trace()
        force = self.contact_force * delta_pos / dist * penetration #contract_force = 100　　ベクトルに変換
        # debug
        """if np.sqrt(np.sum(np.square(force)))>4:
            if not((isinstance(entity_a, Landmark)) and (isinstance(entity_b, Landmark))):
                print("{}  and  {} given {}".format(entity_a.name, entity_b.name, force))"""
        # end
        # agentとagentどうしだと反発力を軽減させる
        if entity_a.movable and entity_b.movable:
            force /= 3          #反発力1/3
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
    
    # bound entity(10/26)
    def bound(self):
        for entity in self.entities:
            for i, pos in enumerate(entity.state.p_pos):
                if pos < -1:
                    entity.state.p_pos[i] = -1 - (entity.state.p_pos[i] - (-1))
                if pos > 1:
                    entity.state.p_pos[i] = 1 - (entity.state.p_pos[i] - 1)