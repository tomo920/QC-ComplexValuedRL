class EnvBase():
    def __init__(self, config, observation_size, action_size, max_step, action_list):
        self.config = config
        self.observation_size = observation_size
        self.action_size = action_size
        self.max_step = max_step
        self.action_list = action_list
        self.reset()

    def discret(self, s, o_list):
        for n in range(1, self.config.n_equal_part+1):
            if s <= o_list[n]:
                return n

    def get_observation(self):
        raise NotImplementedError("You have to implement get_observation method.")

    def reset_state(self):
        raise NotImplementedError("You have to implement reset_state method.")

    def reset(self):
        self.reset_state()
        self.steps = 0
        return self.get_observation()

    def change_state(self, action):
        raise NotImplementedError("You have to implement change_state method.")

    def check_goal(self):
        raise NotImplementedError("You have to implement check_goal method.")

    def step(self, action):
        self.steps += 1
        self.change_state(action)
        if self.check_goal():
            reward = self.config.goal_reward
            done = True
        elif self.steps == self.max_step:
            reward = 0.0
            done = True
        else:
            reward = 0.0
            done = False
        return self.get_observation(), reward, done

    def get_state(self):
        raise NotImplementedError("You have to implement get_state method.")

    def set_state(self, state):
        raise NotImplementedError("You have to implement set_state method.")

    def get_info(self):
        state = self.get_state()
        steps = self.steps
        return [state, steps]

    def load_info(self, env_info):
        state, steps = env_info
        self.set_state(state)
        self.steps = steps
