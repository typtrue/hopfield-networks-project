import numpy as np
import cmath as cm

class HopfieldNetwork:
    """
    Represents a simple Hopfield network where the spins are limited to be +1 or -1
    """
    def __init__(self, shape, *, interactions = None, threshold = None):
        """
        Constructor function
        """
        assert (len(shape) == 2)
        self.savedimages = 0
        self.shape = shape
        self.ishape = (np.prod(self.shape), np.prod(self.shape)) if interactions == None else interactions.shape # shape of interactions matrix
        self.interactions = np.zeros(self.ishape) if interactions == None else interactions
        self.threshold = np.zeros(np.prod(self.shape)) if threshold == None else threshold
        
    def update_async(self, states):
        """
        Updates node spins asynchronously
        """
        interactions = self.interactions
        newstates = states.copy()
        for i in range(len(states)):
            u = self.threshold[i]
            k = np.dot(states, interactions[i,].T)
            newstates[i] = 1 if k >= u else -1
            states = newstates.copy()
        return states

    def update_sync(self, states):
        """
        Updates node spins synchronously
        """
        interactions = self.interactions
        newstates = np.zeros(np.prod(self.shape))
        for i in range(len(states)):
            u = self.threshold[i]
            k = np.dot(states, interactions[i,].T)
            newstates[i] = 1 if k >= u else -1
        return newstates

    def train_network(self, states):
        """
        Trains network on given set of states
        """
        self.savedimages += 1
        n = self.savedimages
        traindata = np.outer(states, states.T)
        np.fill_diagonal(traindata, 0)
        self.interactions = ((n-1) * self.interactions + traindata) / n

    def energy_function(self, states):
        """
        Hamiltonian energy function of current state
        """
        return -((self.interactions @ states) @ states) - np.dot(self.threshold, states)



class HopfieldComplex:
    """
    Represents a Hopfield network where the spins are modelled as lying on the complex unit circle
    """
    def __init__(self, shape, *, interactions = None):
        """
        Constructor function
        """
        assert (len(shape) == 2)
        self.savedimages = 0
        self.shape = shape
        self.ishape = (np.prod(self.shape), np.prod(self.shape)) if interactions == None else interactions.shape # shape of interactions matrix
        self.interactions = np.zeros(self.ishape, dtype=np.float16) if interactions == None else interactions

    def update_async(self, states):
        """
        Updates node spins asynchronously
        """
        interactions = self.interactions
        newstates = states.copy()
        for i in range(len(states)):
            avg = np.dot(states,interactions[i,].T) # gets weighted average spin of connected nodes
            newstates[i] = np.exp(1j*np.angle(avg)) # renormalising
            states = newstates.copy()
        return states

    def update_sync(self, states):
        """
        Updates node spins synchronously
        """
        interactions = self.interactions
        newstates = np.zeros(np.prod(self.shape), dtype=np.complex64)
        for i in range(len(states)):
            avg = np.dot(states,interactions[i,].T) # gets weighted average spin of connected nodes
            newstates[i] = np.complex64(np.exp(1j*np.angle(avg))) # renormalising
        return newstates

    def train_network(self, states):
        """
        Trains network on given set of states
        """
        self.savedimages += 1
        n = self.savedimages
        traindata = np.outer(states, (1/states).T)
        traindata = 1 - abs(np.angle(traindata)) * 2 / np.pi # adjusted learning rule
        np.fill_diagonal(traindata, 0)

        self.interactions = ((n-1) * self.interactions + traindata) / n

    def energy_function(self, states):
        """
        Hamiltonian energy function of current state
        """
        return -((self.interactions @ states) @ states)