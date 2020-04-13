# master class that performs environment interaction and learning
class Master():

    def __init__(self,
                 env,
                 n_actor,
                 n_critic,
                 n_place,
                 dt,
                 V_0,
                 tau_r,
                 stepSize=1,
                 actor_lr=0.05,
                 critic_lr=0.2):

        # gym
        self.env = env
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            self.action_dim = env.action_space.n
            self.discrete_actions = True
        else:
            self.action_dim = env.action_space.shape[0]
            self.discrete_actions = False

        self.state_dim = env.observation_space.shape[0]
        self.stepsize = stepSize
        self.dt = dt
        self.state = env.reset()
        self.reward = 0
        self.done = False
        self.reward_history = []
        self.totalReward = 0

        self.V_0 = V_0
        self.tau_r = tau_r
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.place_actor_weights = 2 * np.random.randn(n_actor,
                                                       n_place) / n_actor
        self.place_critic_weights = 2 * np.random.randn(n_critic,
                                                        n_place) / n_critic
#         self.place_actor_weights = np.random.uniform(n_actor, n_place)
#         self.place_critic_weights = np.random.uniform(n_critic, n_place)
#         self.td_error_history = []

    def step(self, t, x):
        if int(t / self.dt) % self.stepsize != 0:
            return

        if self.discrete_actions:
            action = np.argmax(x)
        else:
            action = x
#         self.env.render()
#         print(f'STEP... x: {x}, action: {action}')
        self.state, self.reward, self.done, _ = self.env.step(action)
        self.totalReward += self.reward
        if self.done:
            #             print('done')
            self.reward = -2
            self.totalReward += self.reward
            self.reward_history.append(self.totalReward)
            self.state = self.env.reset()
            self.totalReward = 0


#     def calc_td_error(self, t, x):
#         td_error = np.sum(x) - self.V_0/self.tau_r + self.reward
#         print('td_error:', td_error, np.sum(x), self.reward)
#         self.td_error_history.append(td_error)
#         return td_error

    def outer(self, t, x):
        X_j_conv_eps = x[:n_place]
        Y = x[n_place:]
        return np.outer(Y, X_j_conv_eps).flatten()

    def actor(self, t, x):
        dVdw = x[:n_place * n_actor].reshape(n_actor, n_place)
        place_spikes = x[n_place * n_actor:-1]
        td_error = x[-1]  #self.td_error_history[-1]
        #         print(f'ACTOR... mean(dVdw): {np.mean(dVdw)}, sum(dVdw){np.sum(dVdw)}, td: {td_error}')
        self.place_actor_weights += self.actor_lr * td_error * dVdw

        return np.dot(self.place_actor_weights, place_spikes)

    def critic(self, t, x):
        dVdw = x[:n_place * n_critic].reshape(n_critic, n_place)
        place_spikes = x[n_place * n_critic:-1]
        td_error = x[-1]  #self.td_error_history[-1]
        #         print(f'CRITIC... mean(dVdw): {np.mean(dVdw)}, sum(dVdw){np.sum(dVdw)}, td: {td_error}')
        self.place_critic_weights += self.critic_lr * td_error * dVdw

        return np.dot(self.place_critic_weights, place_spikes)

V_0 = 0
tau_r = 1  # reward time constant
v_k = 50e-3
tau_k = 200e-3
eps_0 = 20e-3
tau_m = 20e-3
tau_s = 5e-3
dt = 1e-3
n_actor = 100
n_critic = 100
n_place = 100
place_radius = 10
actor_radius = 2
critic_radius = 10
stepSize = 5
actor_lr = 0.05
critic_lr = 0.2

# env = gym.make('CartPole-v0')
# env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')
# env = gym.make('Acrobot-v1')
master = Master(
    env=env,
    n_actor=n_actor,
    n_critic=n_critic,
    n_place=n_place,
    dt=dt,
    V_0=V_0,
    tau_r=tau_r,
    stepSize=stepSize,
    actor_lr=actor_lr,
    critic_lr=critic_lr)
model = nengo.Network()

with model:
    state_node = nengo.Node(output=master.state)
    reward_node = nengo.Node(output=master.reward)
    place = nengo.Ensemble(
        n_neurons=n_place, dimensions=master.state_dim, radius=place_radius)
    nengo.Connection(state_node, place)

    actor = nengo.Ensemble(
        n_neurons=n_actor, dimensions=master.action_dim, radius=actor_radius)
    actor_learn_conn = nengo.Connection(
        place,
        actor,
        function=lambda x: [0] * master.action_dim,
        learning_rule_type=nengo.PES(actor_lr))

    step_node = nengo.Node(output=master.step, size_in=master.action_dim)
    nengo.Connection(actor, step_node)

    critic = nengo.Ensemble(
        n_neurons=n_critic, dimensions=1, radius=critic_radius)
    critic_learn_conn = nengo.Connection(
        place,
        critic,
        function=lambda x: 0,
        learning_rule_type=nengo.PES(critic_lr))

    nengo.Connection(
        reward_node, critic_learn_conn.learning_rule, synapse=0.02, transform=1)
    nengo.Connection(
        critic, critic_learn_conn.learning_rule, synapse=0.02, transform=-1)
    nengo.Connection(
        reward_node,
        actor_learn_conn.learning_rule,
        #                      function=lambda x:[0]*master.action_dim,
        synapse=0.02,
        transform=1)
    nengo.Connection(
        critic,
        actor_learn_conn.learning_rule,
        #                      function=lambda x:[0]*master.action_dim,
        synapse=0.02,
        transform=-1)

    #     td_error_node = nengo.Node(output=None, size_in=1)
    #     nengo.Connection(reward_node, td_error_node, transform=1)
    #     nengo.Connection(critic, td_error_node, transform=-1)
    #     # pass td_error to learn nodes
    #     nengo.Connection(td_error_node,
    #                      actor_learn_conn.learning_rule,
    # #                      function=lambda x:[0]*master.action_dim,
    #                      synapse=0.02)
    #     nengo.Connection(td_error_node,
    #                      critic_learn_conn.learning_rule,
    #                      function=lambda x:list(np.zeros(1)),
    #                      synapse=0.02)

    #     err_probe = nengo.Probe(td_error_node, synapse=None)
    actor_probe = nengo.Probe(actor, synapse=None)
    critic_probe = nengo.Probe(critic, synapse=None)
    place_probe = nengo.Probe(place, synapse=None)

with nengo.Simulator(model) as sim:
    sim.run(20)

master.env.close()