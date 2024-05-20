class my_learner_decay:
    # initialising learner
    def __init__(self, width=19, height=15, v_y=20, v_x=20, n_actions=9, explore=0.1, planning_steps=15, discount=0.95, l_rate=0.6):
        self.explore = explore
        self.planning_steps = planning_steps
        self.discount = discount
        self.l_rate = l_rate
        self.Q_table = np.zeros((height, width, v_y, v_x, n_actions))
        self.model = {}
        self.rewards = 0

    # deciding between choosing greedy and exploring
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.explore:
            return np.random.choice(env.get_actions())
        else:
            return np.argmax(self.Q_table[state[0], state[1], state[2]+10, state[3]+10])
        
    # updating our Q table values using our reward from our action, learn rate, current q value, and discounted next max q value
    def update_Q(self, state, action, reward, next_state):
        max_next_Q = np.max(self.Q_table[next_state[0], next_state[1], next_state[2]+10, next_state[3]+10])
        self.Q_table[state[0], state[1], state[2]+10, state[3]+10][action] += self.l_rate * (reward + (self.discount * max_next_Q) - self.Q_table[state[0], state[1], state[2]+10, state[3]+10][action])
    
    # updating our model from the state action just experience so we can use it to simulate other experiences
    def update_model(self, state, action, reward, next_state):
        self.model[(state, action)] = (reward, next_state)

    # simulating random experiences from our model of stored past state and actions
    def planning(self):
        state, action = [key for key in self.model.keys()][np.random.randint(len(self.model))]
        reward, next_state = self.model[(state, action)]
        self.update_Q(state, action, reward, next_state)

    # setting up the training
    def training(self):
        reward_list = []
        for _ in range(150):
            state = env.reset()
            terminal = False
            self.rewards = 0
            while not terminal:
                action = self.choose_action(state)
                new_state, reward, terminal = env.step(action)
                # add reward for action to total run reward
                self.rewards += reward
                # using functions explained above
                self.update_Q(state, action, reward, new_state)
                self.update_model(state, action, reward, new_state)
                # using planning function explained above for the number of planning steps deemed appropriate
                for _ in range(self.planning_steps):
                    self.planning()
                # moving our agent to their new state
                state = new_state
            reward_list.append(self.rewards)
            # decreasing or learn rate as we go on so that once we have sufficient information we don't change our policy due to randomness
            self.l_rate *= 0.99
            # if function designed to decrease epsilon as our performance levels off so that we explore less on the assumption we are nearing optimum
            if self.rewards != max(reward_list):
                self.explore *= 0.6
        
        return reward_list

modified_agent_rewards = []
# training 25 agents for an average
agents = 25
for _ in range(agents):
    learner = my_learner_decay()
    rewards = learner.training()
    modified_agent_rewards.append(rewards)

plot_modified_agent_results(modified_agent_rewards)
