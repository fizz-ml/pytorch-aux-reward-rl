## Notes 


DDPG with Auxillary Rewards:

### Main Actor-Critic Model:

State -> Actor -> Action  
State, Action -> Critic-> Q(State,Action)


Aux_reward_i:

State -> Actor -> lower level representation of state (LRS)  
LRS -> aux_reward_module_i -> Aux_reward_i 

### To train, backprop:

mean_square_loss(Q , Q_obs) ->  critic -> State, action 

-Q -> critic -> State, action -> Actor -> State 
