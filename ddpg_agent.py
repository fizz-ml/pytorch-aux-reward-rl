import torch
import agent

DEFAULT_REPLAY_BUFFER_SIZE = 1000000
DEFAULT_DISCOUNT_FACTOR = 1
DEFAULT_LEARNING_RATE = 0.01

class DDPGAgent(agent.Agent):
    """An agent that implements the DDPG algorithm

        An agent that implements the deep deterministic
        policy gradient algorithm for continuous control.
        A description of the algorithm can be found at
        https://arxiv.org/pdf/1509.02971.pdf.

        The agent stores a replay buffer along with
        two models of the data, an actor and a critic.

        Attributes:
            auxillary_rewards: The list of enabled
            auxillary rewards for this agent

            actor: The actor model that takes a state
            and returns a new reward.

            critic: The critic model that takes a state
            and an action and returns a new 

            replay_buffer: The DDPGAgent replay buffer
    """
    self._target_critic=None
    self._alpha = None
    self._gamma = None
    self._actor_location = None
    self._critic_location = None

    @property
    def actor():
        return self.actor()
        pass

    @property
    def critic():
        return self.critic()
        pass

    @property
    def replay_buffer():
        return self.replay_buffer()

    def __init__(actor_location, critic_location,
            buffer_size=DEFAULT_REPLAY_BUFFER_SIZE,
            gamma=DEFAULT_DISCOUNT_FACTOR,
            alpha=DEFAULT_LEARNING_RATE):
        """Constructor for the DDPG_agent

            Args:
                actor_location: location of the actor_t7

                critic_location: location of the critic_t7

                buffer_size: size of the replay buffer

                alpha: The learning rate
                
                gamma: The discount factor 
                
            Returns:
                A DDPGAgent object
        """
    
    def train():
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        raise NotImplementedError

    def get_next_action(cur_state,
            prev_reward,
            is_done=False,
            agent_id=None):
        """Get the next action from the agent.
            
            Takes a state,reward and possibly auxiliary reward
            tuple and returns the next action from the agent.
            The agent may cache the reward and state 

            Args:
                cur_state: The current state of the enviroment
                prev_reward: The previous reward from the enviroment
                is_done: Signals if a given episode is done.
                agent_id=None
            Returns:
                The next action that the agent with the given 
                agent_id will carry out given the current state
        """
        raise NotImplementedError

    def save_models(locations=None):
        """Save the model to a given locations

            Args:
                Locations: where to save the model
            Returns:
                None
        """
        raise NotImplementedError
        
    def load_models(locations=None):
        """Loads the models from given locations

            Args:
                Locations: from where to load the model
            Returns:
                None
        """
        raise NotImplementedError

