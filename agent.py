import torch

class Agent:
    """The generic interface for an agent.

        Attributes:
            auxillary_rewards: The list of enabled
            auxillary rewards for this agent
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

    @property
    def auxiliary_rewards():

class DDPGAgent(Agent):
    pass

