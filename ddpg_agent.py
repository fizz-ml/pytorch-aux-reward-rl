import copy
import torch
import torch.optim as opt
import agent

from torch.utils.serialization import load_lua

#Default hyperparameter values
REPLAY_BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 1
LEARNING_RATE_CRITIC = 0.01
LEARNING_RATE_ACTOR = 0.01
ACTOR_ITER_COUNT = 1000
CRITIC_ITER_COUNT = 1000
BATCH_SIZE = 100


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
    @property
    def actor(self):
        return self.actor

    @property
    def critic(self):
        return self.critic

    @property
    def replay_buffer(self):
        return self.replay_buffer

    def __init__(self,actor_location, critic_location,
            buffer_size=REPLAY_BUFFER_SIZE,
            gamma=DISCOUNT_FACTOR,
            actor_alpha=LEARNING_RATE_ACTOR
            critic_alpha=LEARING_RATE_CRITIC
            actor_iter_count=ACTOR_ITER_COUNT
            critic_iter_count=CRITIC_ITER_COUNT
            batch_size=BATCH_SIZE,
            auxillary_loss_modules={}):
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
        #Initialize experience replay buffer
        self.replay_buffer = #TODO

        #initialize parameters
        self._actor_alpha = actor_alpha
        self._critic_alpha = critic_alpha
        self._actor_iter_count = actor_iter_count
        self._critic_iter_count = critic_iter_count
        self._gamma = gamma
        self._batch_size = batch_size
        
        #Specify model locations
        self._actor_location = actor_location
        self._critic_location = critic_location

        #Initialize optimizers
        self._actor_optimizer = opt.ADAM(actor.parameters(), lr=_self._actor_alpha)
        self._critic_optimizer = opt.ADAM(critic.parameters(), lr=_self._critic_alpha)

        
        #initialize models
        self.load_models()
        self._target_critic = self.critic

    
    def train(self):
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        #update_critic
        for i in range(self._critic_iter_count)
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            a_t1 = self.actor.forward(s_t1,r_t,[])
            critic_target = r_t + self._gamma*(1-done)*self._target_critic.forward(s_t1,a_t1)
            td_error = (self.critic.forward(s_t,a_t)-critic_target)**2
            
            #preform one optimization update
            _critic_optimizer.zero_grad()
            td_error.backwards()
            _critic_optimizer.step()


        #update_actor
        for i in range(self._actor_iter_count)
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            a_t1,aux_actions = self.actor.forward(s_t1,r_t,self.auxiliary_rewards.keys())
            expected_reward = self.critic.forward(s_t1,a_t1)
            
            total_loss = -1*expected_reward
            for key,aux_reward_tuple in self.auxillary_rewards.items()
                aux_weight,aux_module = aux_reward_tuple
                total_loss += aux_weight*aux_module(aux_actions[key])

            loss = torch.sum(total_loss)

            #preform one optimization update
            _actor_optimizer.zero_grad()
            loss.backwards()
            _actor_optimizer.step()

        
        


    def get_next_action(self,
            cur_state,
            prev_reward,
            is_done=False,
            agent_id=None
            is_test=False):
        """Get the next action from the agent.
            
            Takes a state,reward and possibly auxiliary reward
            tuple and returns the next action from the agent.
            The agent may cache the reward and state 

            Args:
                cur_state: The current state of the enviroment
                prev_reward: The previous reward from the enviroment
                is_done: Signals if a given episode is done.
                is_test: Check to see if the agent is done
                agent_id=None
            Returns:
                The next action that the agent with the given 
                agent_id will carry out given the current state
        """
        if (prev_reward != None):
            self.replay_buffer.put_rew(prev_reward,is_done)
        cur_action = self.actor.forwards(cur_state,prev_reward,[])
        self.replay_buffer.put_act(cur_state,cur_action)
        
        return cur_action

    def save_models(self, locations=None):
        """Save the model to a given locations

            Args:
                Locations: where to save the model
            Returns:
                None
        """
        #Return all weights and buffers to the cpu
        self.actor.cpu()
        self.critic.cpu()

        #Save both models
        torch.save(self._actor_location,actor)
        torch.save(self._actor_location,critic)
        
    def load_models(self, locations=None):
        """Loads the models from given locations

            Args:
                Locations: from where to load the model
            Returns:
                None
        """
        self.actor = torch.load_lua(self._actor_location)
        self.critic = torch.load_lua(self._critic_location)

        #Move weights and bufffers to the gpu if possible
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()

