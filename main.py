import config
from environment import GymEnvironment
from ddpg_agent import DDPGAgent
import numpy as np

def main():
    conf = config.config_dict
    run = Runner(conf['env'], conf['agent'])
    run.train(conf['train'])
    run.test(conf['test'])

class Runner:
    def __init__(self, env_config, agent_config):
        self.env = GymEnvironment(**env_config)
        self.agent = DDPGAgent(action_size = self.env.action_size[0],
                                state_size = self.env.obs_size[0],
                                **agent_config) 

    def train(self, train_config, fill_replay = True):
        # Fill experience replay
        if fill_replay:
            prefill = train_config['prefill']
            
            temp_reward = None 
            temp_done = False
            for step in range(prefill):
                cur_obs = self.env.cur_obs
                cur_action = self.agent.get_next_action(cur_obs, temp_reward, temp_done)

                next_state, reward, done = self.env.next_state(cur_action, render = True) 

                temp_reward = reward
                temp_done = done 

        # Start training
        train_steps = train_config['steps']

        temp_reward = None 
        temp_done = False 
        for step in range(train_steps):
            cur_obs = self.env.cur_obs
            cur_action = self.agent.get_next_action(cur_obs, temp_reward, temp_done)

            next_state, reward, done = self.env.next_state(cur_action, render = True) 

            temp_reward = reward
            temp_done = done

            agent.train()

    def test(self, test_config):
        test_steps = test_config['steps']
 
        temp_reward = None 
        temp_done = False       
        for step in range(start_train):
            cur_obs = self.env.cur_obs
            cur_action = self.agent.get_next_action(cur_obs, temp_reward, temp_done)
            next_state, reward, done = self.env.next_state(cur_action, render = True) 

if __name__ == "__main__":
    main()
