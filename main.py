import config

def main():
    conf = config.config_dict
    run = Runner(conf['env'], conf['agent'])
    run.train(conf['train'])
    run.test(conf['test'])

class Runner:
    def __init__(env_config, agent_config):
        self.env = GymEnvironment('CartPole')   
        self.agent = DDPGAgent(actor_path, critic_path, **agent_config) 

    def train(train_config, fill_replay = True):
        # Fill experience replay
        if fill_replay:
            prefill = train_config['prefill']
            
            temp_reward = 0
            temp_done = False
            for step in xrange(prefill):
                cur_state = self.env.cur_obs
                cur_action = self.agent.get_next_action(cur_obs, temp_reward, temp_done)

                next_state, reward, done = self.env.next_state(cur_action, render = True) 

                temp_reward = reward
                temp_done = done 

        # Start training
        train_steps = train_config['steps']

        temp_reward = 0
        temp_done = False 
        for step in xrange(train_steps):
            cur_state = self.env.cur_obs
            cur_action = self.agent.get_next_action(cur_obs, temp_reward, temp_done)

            next_state, reward, done = self.env.next_state(cur_action, render = True) 

            temp_reward = reward
            temp_done = done

            agent.train()

    def test(test_config):
        test_steps = test_config['steps']
        
        for step in xrange(start_train):
            cur_state = self.env.cur_obs
            cur_action = self.agent.get_next_action(cur_obs, temp_reward, temp_done)
            next_state, reward, done = self.env.next_state(cur_action, render = True) 

if __name__ == "__main__":
    main()
