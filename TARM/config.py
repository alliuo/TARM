import os
import yaml


class Config:
    """
    Configuration of one run. Specify the CNN application and constraints
    """
    def __init__(self, config_file="./config/config.yml"):
        self.config_file = config_file # the path to the config file
        self.abc, self.tfapprox_path, self.application, self.application_path, self.dist_file, self.ME_constraint, self.accuracy_threshold = self._load_configs()
    

    def _load_configs(self):
        """
        Load configuration for each application
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                abc = os.path.abspath(config['abc'])
                tfapprox_path = os.path.abspath(config['tfapprox_path'])
                application = config['application']
                application_path = os.path.abspath(config['application_path'])
                dist_file = os.path.abspath(config['dist_file'])
                ME_constraint = config['ME_constraint']
                accuracy_threshold = config['accuracy_threshold']
        else:
            raise Exception("Config file not found")
        return abc, tfapprox_path, application, application_path, dist_file, ME_constraint, accuracy_threshold
