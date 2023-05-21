class IterMeter(object):
    """Keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def train():
    pass


def test():
    pass