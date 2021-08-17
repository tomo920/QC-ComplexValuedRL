class QlstmBase():
    def __init__(self, config, env):
        self.config = config
        self.env = env

    def get_q_hi(self, hidden_state, input_seq):
        raise NotImplementedError("You have to implement get_q_hi method.")

    def get_q(self, hidden_state, input_seq):
        q = self.get_q_hi(hidden_state, input_seq)
        if self.config.is_legal_action:
            return q[self.env.legal_action_list[observation]]
        return q

    def get_effective_q(self, hidden_state, input_seq):
        return self.get_q(hidden_state, input_seq)

    def update_q(self, input_seq, action_seq, target_seq):
        raise NotImplementedError("You have to implement update_q method.")
