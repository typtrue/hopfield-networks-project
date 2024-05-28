import numpy as np
import cmath as cm

class HopfieldNetwork:

    def __init__(self, shape, *, interactions = None, threshold = None):
        assert (len(shape) == 2)
        self.savedimages = 0
        self.shape = shape
        self.ishape = (np.prod(self.shape), np.prod(self.shape)) if interactions == None else interactions.shape
        self.interactions = np.zeros(self.ishape) if interactions == None else interactions
        self.threshold = np.zeros(np.prod(self.shape)) if threshold == None else threshold
        
    def update_async(self, states):
        interactions = self.interactions
        newstates = states.copy()
        for i in range(len(states)):
            u = self.threshold[i]
            k = np.dot(states, interactions[i,].T)
            newstates[i] = 1 if k >= u else -1
            states = newstates.copy()
        return states

    def update_sync(self, states):
        interactions = self.interactions
        newstates = np.zeros(np.prod(self.shape))
        for i in range(len(states)):
            u = self.threshold[i]
            k = np.dot(states, interactions[i,].T)
            newstates[i] = 1 if k >= u else -1
        return newstates

    def train_network(self, states):
        self.savedimages += 1
        n = self.savedimages
        traindata = np.outer(states, states.T)
        np.fill_diagonal(traindata, 0)
        self.interactions = ((n-1) * self.interactions + traindata) / n

    def energy_function(self, states):
        return -((self.interactions @ states) @ states) - np.dot(self.threshold, states)



class HopfieldComplex:

    def __init__(self, shape, *, interactions = None):
        assert (len(shape) == 2)
        self.savedimages = 0
        self.shape = shape
        self.ishape = (np.prod(self.shape), np.prod(self.shape)) if interactions == None else interactions.shape
        self.interactions = np.zeros(self.ishape, dtype=np.float16) if interactions == None else interactions

    # def update_async(self, states):
    #     interactions = self.interactions
    #     newstates = states.copy()
    #     for i in range(len(states)):
    #         avg = np.dot(states,interactions[i,].T)
    #         newstates[i] = np.exp(1j*np.angle(avg))
    #         states = newstates.copy()
    #     return states

    # def update_sync(self, states):
    #     interactions = self.interactions
    #     newstates = np.zeros(np.prod(self.shape), dtype=np.complex64)
    #     for i in range(len(states)):
    #         avg = np.dot(states,interactions[i,].T)
    #         newstates[i] = np.complex64(np.exp(1j*np.angle(avg)))
    #     return newstates

    # def train_network(self, states):
    #     self.savedimages += 1
    #     n = self.savedimages
    #     traindata = np.outer(states, (1/states).T)
    #     traindata = 1 - abs(np.angle(traindata)) * 2 / np.pi
    #     np.fill_diagonal(traindata, 0)

    #     self.interactions = ((n-1) * self.interactions + traindata) / n

    def update_async(self, states):
        interactions = self.interactions
        newstates = states.copy()
        for i in range(len(states)):
            avg = np.dot(states,interactions[i,].T) / states.size
            newstates[i] = avg
            states = newstates.copy()
        return states

    def update_sync(self, states):
        interactions = self.interactions
        newstates = np.zeros(np.prod(self.shape), dtype=np.float32)
        for i in range(len(states)):
            avg = np.dot(states,interactions[i,].T) / states.size
            print(avg)
            newstates[i] = avg
        return newstates

    def train_network(self, states):
        self.savedimages += 1
        n = self.savedimages
        k = np.tile(states, (states.size, 1))
        traindata = 1 - abs(k - k.T)

        np.fill_diagonal(traindata, 0)


        a = np.ones((states.size,)) - np.identity(states.size)

        self.interactions = ((n-1) * self.interactions + traindata) / n

    def energy_function(self, states):
        return -((self.interactions @ states) @ states)