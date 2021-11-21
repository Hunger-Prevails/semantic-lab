import math

class Adapter:

    def __init__(self, args, optimizer, n_batches):
        self.optimizer = optimizer
        self.n_batches = n_batches

        self.n_epochs = args.n_epochs
        self.n_iters_start = args.n_iters_start

        self.learn_rate = args.learn_rate
        self.learn_rate_start = args.learn_rate_start
        self.learn_rate_decay = args.learn_rate_decay
        self.learn_rate_haven = args.learn_rate_haven


    def on_start(self, state):
        factor = (self.learn_rate_start - 1.0) / self.n_iters_start ** 2

        return factor * min(state['past_iters'] - self.n_iters_start, 0) ** 2 + 1.0


    def percent(self, state):
        return max(state['past_iters'] - self.n_iters_start, 0) / (self.n_epochs * self.n_batches - self.n_iters_start)


    def on_batch(self, state):
        raise NotImplementedError()


    def schedule(self, state):
        current = self.on_batch(state) * self.on_start(state)

        for group in self.optimizer.param_groups:
            group['lr'] = current


class PolynomAdapter(Adapter):

    def __init__(self, args, optimizer, n_batches):
        super().__init__(args, optimizer, n_batches)


    def on_batch(self, state):
        factor = (1 - self.percent(state)) ** self.learn_rate_decay

        return self.learn_rate_haven + (self.learn_rate - self.learn_rate_haven) * factor


class CosineAdapter(Adapter):

    def __init__(self, args, optimizer, n_batches):
        super().__init__(args, optimizer, n_batches)


    def on_batch(self, state):
        factor = (1 + math.cos(math.pi * self.percent(state))) / 2

        return self.learn_rate_haven + (self.learn_rate - self.learn_rate_haven) * factor
