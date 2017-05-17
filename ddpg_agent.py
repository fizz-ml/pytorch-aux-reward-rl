import torch
import torch.optim as opt
import agent
from replay_buffer import ExperienceReplay
import numpy as np
import dill
import random
from torch.utils.serialization import load_lua
import model_defs.ddpg_models.mountain_cart.critic as critic
import model_defs.ddpg_models.mountain_cart.actor as actor
import random

#Default hyperparameter values
REPLAY_BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 1
LEARNING_RATE_CRITIC = 0.01
LEARNING_RATE_ACTOR = 0.01
ACTOR_ITER_COUNT = 1000
CRITIC_ITER_COUNT = 1000
BATCH_SIZE = 100
EPSILON = 0.01

class DDPGAgent(agent.Agent):
    """An agent that implements the DDPG algorithm

    An agent that implements the deep deterministic
    policy gradient algorithm for continuous control.
    A description of the algorithm can be found at
    https://arxiv.org/pdf/1509.02971.pdf.

    The agent stores a replay buffer along with
    two models of the data, an actor and a critic.

    Attributes:
        auxiliary_losses: The list of enabled
        auxiliary rewards for this agent

        actor: The actor model that takes a state
        and returns a new action.

        critic: The critic model that takes a state
        and an action and returns the expected 
        reward

        replay_buffer: The DDPGAgent replay buffer
    """

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
    """

    def __init__(self,actor_path, critic_path,
            state_size = 1,
            action_size = 1,           
            buffer_size = REPLAY_BUFFER_SIZE,
            gamma = DISCOUNT_FACTOR,
            actor_alpha = LEARNING_RATE_ACTOR,
            critic_alpha = LEARNING_RATE_CRITIC,
            actor_iter_count = ACTOR_ITER_COUNT,
            critic_iter_count = CRITIC_ITER_COUNT,
            batch_size = BATCH_SIZE,
            auxiliary_losses = {}):
        """Constructor for the DDPG_agent

        Args:
            actor_path: location of the actor_t7

            critic_path: location of the critic_t7

            buffer_size: size of the replay buffer

            alpha: The learning rate
            
            gamma: The discount factor 
            
        Returns:
            A DDPGAgent object
        """
        super(DDPGAgent, self).__init__(auxiliary_losses)
        
        #Initialize experience replay buffer
        print(state_size)
        self.replay_buffer = ExperienceReplay(state_size, action_size, buffer_size)
        #TODO

        #initialize parameters
        self._actor_alpha = actor_alpha
        self._critic_alpha = critic_alpha
        self._actor_iter_count = actor_iter_count
        self._critic_iter_count = critic_iter_count
        self._gamma = gamma
        self._batch_size = batch_size
        
        #Specify model locations
        self._actor_path = actor_path
        self._critic_path = critic_path
        
        #initialize models
        self.load_models()

        #Initialize optimizers
        self._actor_optimizer = opt.Adam(self.actor.parameters(), lr=self._actor_alpha)
        self._critic_optimizer = opt.Adam(self.critic.parameters(), lr=self._critic_alpha)

            
    def train(self):
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        #update_critic
        for i in range(self._critic_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            a_t1 = self.actor.forward(s_t1,r_t,[])
            critic_target = r_t + self._gamma*(1-done)*self._target_critic.forward(s_t1,a_t1)
            td_error = (self.critic.forward(s_t,a_t)-critic_target)**2
            
            #preform one optimization update
            _critic_optimizer.zero_grad()
            td_error.backwards()
            _critic_optimizer.step()


        #update_actor
        for i in range(self._actor_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            a_t1,aux_actions = self.actor.forward(s_t1,r_t,self.auxiliary_losses.keys())
            expected_reward = self.critic.forward(s_t1,a_t1)
            
            total_loss = -1*expected_reward
            for key,aux_reward_tuple in self.auxiliary_losses.items():
                aux_weight,aux_module = aux_reward_tuple
                total_loss += aux_weight*aux_module(aux_actions[key],s_t,a_t,r_t,s_t1,a_t1)

            loss = torch.sum(total_loss)

            #preform one optimization update
            _actor_optimizer.zero_grad()
            loss.backwards()
            _actor_optimizer.step()
        
        # TODO: Freeze less often
        self._target_critic.load_state_dict(self.critic.state_dict())
        


    def get_next_action(self,
            cur_state,
            prev_reward,
            agent_id=None,
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
        cur_action = None
        cur_action = self.actor.forward(cur_state, np.expand_dims(prev_reward, axis = 0),[]).data.cpu().numpy()
        self.replay_buffer.put_act(cur_state,cur_action)
        return cur_action

    def log_reward(prev_reward,is_done):
            self.replay_buffer.put_rew(prev_reward,is_done)

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
        actor_file=open(self._actor_path,"wb")
        dill.dump(self.actor,actor_file)
        critic_file=open(self._critic_path,"wb")
        dill.dump(self.critic,critic_file)
        
    def load_models(self, locations=None):
        # TODO: Make it actually do what it says
        """Loads the models from given locations

            Args:
                Locations: from where to load the model
            Returns:
                None
        """
        actor_file=open(self._actor_path,"rb")
        print(self._actor_path)
        print(actor_file)
        self.actor = actor.Actor(3,1) #dill.load(actor_file)
        critic_file=open(self._critic_path,"rb")
        self.critic = critic.Critic(3,1)#dill.load(critic_file)
        self._target_critic = critic.Critic(3,1)#dill.load(critic_file)

        #Move weights and bufffers to the gpu if possible
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
            self._target_critic()

