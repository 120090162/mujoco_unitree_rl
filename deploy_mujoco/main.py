from deploy import *

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="config file name in the config folder"
    )
    args = parser.parse_args()
    config_file = args.config_file
    rl_sim = RL_Sim(config_file=config_file)
    rl_sim.main_loop()
