config_dict = dict(
agent   =   {
            'replay_size' : 100000 
            },

train   =   {
            'steps' : 500000
            'prefill' : 100000
            },

test    =   {
            'steps' : 10000
            },

env     =   {
            'name' : 'CartPole'
            },

auxw    =   {
            'base' : 2
            'reward_predict' : 1
            }
)
