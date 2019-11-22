import numpy as np
import random

class QLearner():

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.experiences = np.empty((0, 4))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # action = rand.randint(0, self.num_actions-1)
        # if self.verbose: print "s =", s,"a =",action
        if rand.random() > self.rar:
            action = np.argmax(self.Q[self.s, :])
        else:
            action = rand.randint(0, self.num_actions - 1)
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        def updateQ(s, a, s_prime, r):
            if rand.random() > self.rar:
                action = np.argmax(self.Q[s_prime, :])
            else:
                action = rand.randint(0, self.num_actions - 1)
            old_Q = (1 - self.alpha) * self.Q[s, a]
            self.Q[s, a] =  old_Q + self.alpha * (r + self.gamma * self.Q[s_prime, action])
            return action
        if self.dyna != 0:
            self.experiences = np.vstack((self.experiences, [self.s, self.a, s_prime, r]))
            if len(self.experiences.shape) == 1:
                self.experiences = self.experiences.reshape((1, 4))
            num_experiences = self.experiences.shape[0]
            if num_experiences < self.dyna:
                samples = np.tile(self.experiences, (self.dyna/num_experiences + 1, 1))[:self.dyna, :]
            else:
                samples = self.experiences[np.random.choice(num_experiences, self.dyna, replace=False), :]
            for row in samples:
                updateQ(int(row[0]), int(row[1]), int(row[2]), row[3])
        action = updateQ(self.s, self.a, s_prime, r)
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action
        return action
