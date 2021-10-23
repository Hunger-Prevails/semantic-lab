class Adapter:

    def __init__(self, optimizer, args):
        self.optimizer = optimizer

        self.n_epochs = args.n_epochs
        self.n_iters_start = args.n_iters_start

        self.learn_rate = args.learn_rate
        self.learn_rate_start = args.learn_rate_start
        self.learn_rate_decay = args.learn_rate_decay
        self.learn_rate_bottom = args.learn_rate_bottom

    def start(self, state):
        factor = (self.learn_rate_start - self.learn_rate) / self.n_iters_start ** 2

        return factor * min(state['past_iters'] - self.n_iters_start, 0) ** 2 / self.learn_rate + 1

    def by_epoch(self, state):
        return self.learn_rate * (1 - state['past_epochs'] / self.n_epochs) ** self.learn_rate_decay

    def schedule(self, state):
        current = max(self.by_epoch(state), self.learn_rate_bottom) * self.start(state)

        for group in self.optimizer.param_groups:
            group['lr'] = current
