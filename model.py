import torch
from torch.autograd import Variable
from models import config


class generator():
"""generate actor, critic models as .t7 files"""
    def _init_(self, name, spec_file, save_location, is_actor = True, dim_input, dim_output):
        self._name = name
        self._spec_file = spec_file
        self._save_location = save_location 
        self.is_actor = is_actor       
        self._dim_input = dim_input
        self._dim_output = dim_output
        
    def generate_model(self):
        """create and save the specified model as .t7 file"""

        model = config.model(self._dim_input, self._dim_output)
        



    def save_model(self):
        torch.save(self._save_location +'/'+ self._name+'actor.t7', self._actor_model);
        torch.save(self._save_location + '/' + self._name+'critic.t7',  self._critic_model);



def main(argv):
    opts, args = getopt.getopt(argv,"nfsa")
    for opt, arg in opts:
        if opt == "n":
            name = arg
        elif opt == "f":
            spec_file = arg
        elif opt == "s":
            save_location = s
        elif opt == "a":
            is_actor = arg

    gen = generator(name, spec_file, save_location, is_actor)
    gen.generate_model()


if __name__ == "main":
    main(sys.argv[1:])








