import numpy as np
import random

class ReplayBuffer:
    """ Buffer for storing values over timesteps.
    """
    def __init__(self):
        """ Initializes the buffer.
        """
        pass

    def batch_sample(self, batch_size):
        """ Randomly sample a batch of values from the buffer.
        """
        raise NotImplementedError

    def put(self, *value):
        """ Put values into the replay buffer. 
        """
        raise NotImplementedError

class ExperienceReplay:
    '''
    Experience Replay stores action states and  
    '''
    def __init__(self, state_size, length):
        # use a current write index implementation for the circular buffer
        self.state_size = state_size
        self.action_size = action_size
        self.length  = length 
        
        self.actions = np.empty((self.length, self.action_size), dtype = np.uint8) 
        self.states = np.empty((self.length, self.state_size), dtype = np.float16)
        self.rewards = np.empty(self.length, dtype = np.float16)
        self.dones = np.empty(self.length, dtype = np.bool)
        
        self.current_index = 0
        
    def batch_sample(self, batch_size):
        idxs = [] 
        while len(idxs) < batch_size:
            while True:
                # keep trying random indices
                idx = random.randint(1, self.length - 1) 
                # don't want to grab current index since it wraps 
                if idx == self.current_index and idx == self.current_index - 1:
                    continue 
                idxs.append(idx)
                break
        s_t = self.states[[(x-1) for x in idxs]]
        s_t1 = self.states[idxs]
        a_t = self.actions[idxs]
        r_t = self.rewards[idxs]
        done = self.dones[idxs]

        '''
        j = 0
        print(s_t[j], s_t1[j], a_t[j], r_t[j], done[j])
        j = 1
        print(s_t[j], s_t1[j], a_t[j], r_t[j], done[j])
        raw_input("Press Enter to continue...")
        '''

        return s_t, a_t, r_t, s_t1, done

    def put(self, s_t, a_t, reward, done):
        self.actions[self.current_index] = a_t 
        self.states[self.current_index] = s_t 
        self.rewards[self.current_index] = reward 
        self.dones[self.current_index] = done
        self.current_index = (self.current_index + 1) % self.length 
