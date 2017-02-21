import torch
from torch.autograd import Variable
from models.config import model as M
import os
import sys
import getopt


class generator():
    """generate actor, critic models as .t7 files"""
    def __init__(self, name, dim_input, dim_output,spec_file,   save_location,is_actor = True):
        
        self._name = name
        self._spec_file = spec_file
        self._save_location = save_location 
        self.is_actor = is_actor       
        self._dim_input = dim_input
        self._dim_output = dim_output
        
    def generate_model(self):
        """create and save the specified model as .t7 file"""     
        
        model = M(self._dim_input, self._dim_output)
        if(self.is_actor):
            
            torch.save(model, os.path.join(self._save_location,'models/actor_models/',self._name+'_actor.t7')) 
        else:
            torch.save(model, os.path.join(self._save_location,'models/critic_models/',self._name+'_critic.t7'))


def main(argv):
    """Specify input to generator with:
    -n : name
    -f : filename
    -s : spec file
    -a : boolean integer specifying whether model is generator or actor
    -i : input dimensionality
    -o : output dimensionality
    """

    opts, args = getopt.getopt(argv,"n:f:s:a:i:o:")
    is_actor = True
    save_location = ""
    name = ""
    spec_file = ""
    dim_input = 100
    dim_output = 1

    print(opts)
    for opt, arg in opts:

        if opt == "-n":
            name = arg
        elif opt == "-f":
            spec_file = arg
        elif opt == "-s":
            save_location = arg
        elif opt == "-a":
            is_actor = bool(arg)
        elif opt == "-i":
            dim_input = int(arg)
        elif opt == "-o":
            dim_output = int(arg)

    gen = generator(name, dim_input,dim_output,spec_file, save_location,is_actor)
    gen.generate_model()


if __name__ == "__main__":
    main(sys.argv[1:])








