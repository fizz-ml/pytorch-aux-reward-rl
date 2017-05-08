import os
import sys
import getopt
import json

def main(argv):
    """Specify input to generator with:
    -s : save path 
    -f : model_def folder 
    """

    opts, args = getopt.getopt(argv,"s:f:")
    save_location = "models/ddpg_models/"
    model_def_folder = ""

    print(opts)
    for opt, arg in opts:
        if opt == "-s":
            save_location = arg
        elif opt == "-f":
            model_def_folder = arg

    json_data = open(os.path.join(model_def_folder,'config.json')).read() 
    config_dict = json.loads(json_data)
    
    print(config_dict)

    exec('')
    os.system("script2.py 1")

if __name__ == "__main__":
    main(sys.argv[1:])
