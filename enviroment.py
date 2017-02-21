import gym

class Environment:
    """ Defines an environment the actor interacts with.
    """

    def __init__(self):
        """ Initializes an environment.
        """
        pass
    
    def next_obs(self, cur_action):
        """ Takes an action in the environment.
         
        """       
        raise NotImplementedError
    
    def new_episode(self):
        """ Starts a new episode of the environment.  
        """
        raise NotImplementedError

    @property
    def action_shape(self):
        """ Returns the shape of the action.
        """
        raise NotImplementedError

    @property
    def obs_shape(self):
        """ Returns the shape of the observation.
        """ 
        raise NotImplementedError


class GymEnvironment(Environment):
    def __init__(self, name):
        """ Initializes a gym environment.
        """
        self.env = gym.make(name)
        self.cur_obs = None

    def next_obs(self, action, render = False):
        """ Runs a step in the gym environment.
        Args:
            action:         Current action to perform 
            render:         (Optional) Wether to render environment or not.
        
        Returns:
            obs:            State of the environment after step.
            reward:         Reward received from the step. 
            done:           Bool signaling terminal step.
        """
        self.env.step(cur_action)
        self.env.step(cur_action)
        self.cur_obs, self.cur_reward, self.done, _ = self.env.step(cur_action)
        if render:
            self.env.render() 
        return self.cur_obs, self.cur_reward, self.done

    def new_episode(self):
        """ Initiates a new episode by resetting the environment.
        Returns:
            obs:    Initial observation of the new episode.
        """
        self.cur_obs = self.env.reset()
        self.env.render() 
        return self.cur_obs

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def obs_size(self):
        return self.cur_obs.shape

    @property
    def cur_obs(self):
        return self.cur_obs 
