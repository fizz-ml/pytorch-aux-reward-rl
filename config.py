config_dict = dict(
    agent   =   dict(
                actor_path          = './models/ddpg_models/ddpg_actor.t7', 
                critic_path         = './models/ddpg_models/ddpg_critic.t7',
                buffer_size         = 1000000,
                gamma               = 0.95,
                actor_alpha         = 0.001,
                critic_alpha        = 0.001,
                actor_iter_count    = 1,
                critic_iter_count   = 1,
                batch_size          = 32,
                auxiliary_losses    =
                    {
                        #'reward_predict' : 1
                    } 
                ),

    train   =   {
                'steps' : 500000,
                'prefill' : 100000
                },

    test    =   {
                'steps' : 10000
                },

    env     =   {
                'name' : 'MountainCarContinuous-v0'
                }
)
