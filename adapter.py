class Adapter:

    def __init__(self, args, optimizer, n_batches):
        self.optimizer = optimizer
        self.n_batches = n_batches

        self.n_epochs = args.n_epochs
        self.n_iters_start = args.n_iters_start

        self.learn_rate = args.learn_rate
        self.learn_rate_start = args.learn_rate_start
        self.learn_rate_decay = args.learn_rate_decay
        self.learn_rate_bottom = args.learn_rate_bottom


    def on_start(self, state):
        factor = (self.learn_rate_start - 1.0) / self.n_iters_start ** 2

        return factor * min(state['past_iters'] - self.n_iters_start, 0) ** 2 + 1.0


    def on_batch(self, state):
        return self.learn_rate * (1 - max(state['past_iters'] - self.n_iters_start, 0) / self.n_epochs / self.n_batches) ** self.learn_rate_decay


    def schedule(self, state):
        current = max(self.on_batch(state), self.learn_rate_bottom) * self.on_start(state)

        for group in self.optimizer.param_groups:
            group['lr'] = current
