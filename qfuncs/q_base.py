class QBase():
    def __init__(self, config, env):
        self.config = config
        self.env = env

    def get_q_o(self, observation):
        raise NotImplementedError("You have to implement get_q_o method.")

    def get_q(self, observation):
        q = self.get_q_o(observation)
        if self.config.is_legal_action:
            return q[self.env.legal_action_list[observation]]
        return q

    def get_effective_q(self, observation):
        return self.get_q(observation)

    def update_q(self, observation, action, q_target):
        raise NotImplementedError("You have to implement update_q method.")

    def get_params(self):
        raise NotImplementedError("You have to implement get_params method.")

    def set_params(self, params):
        raise NotImplementedError("You have to implement set_params method.")
