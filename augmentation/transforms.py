from torchvision import transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ret):
        for transform in self.transforms:
            transform(ret)


class Normalize(object):
	def __init__(self, mean, stddvn):
		self.functor = transforms.Normalize(mean, stddvn)

	def __call__(self, ret):
		ret['image'] = self.functor(ret['image'])
