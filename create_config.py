import os

import yaml
from ml_collections import ConfigDict


def create_config_dict():
    config_dict = ConfigDict(dict(
        transE_embed_dim=100,
        state_dim=200,
        eps_start=1,
        eps_end=0.1,
        epe_decay=1000,
        replay_memory_size=10000,
        batch_size=128,
        gamma=0.99,
        target_update_freq=1000,
        max_steps=50,
        max_steps_test=50,
        num_episods=2000,
        device='cuda:0',
        task='supervised',
        kg_dataset=None,
        train_transE=True,
        save_weights_path='weights',
        normalize_transE_weights=True,
        num_generated_episodes=20,
        num_supervised_epochs=2,
        max_num_examples=-1,
        max_supervised_steps=1000000,
        transE_weights_saved_name='transE_weights.pt',
        dataset_txt_file_path=None,
        rl_phase_load_from_checkpoint=None,
    ))

    return config_dict


def create_arg_parser():
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument('--overwrite', default=False, type=bool)
    return argparser


def write_config_file(config_file_name):
    config_dict = create_config_dict()
    with open(config_file_name, 'w') as f:
        yaml.dump(config_dict, f)
    print(f'{config_file_name} created.')


def to_yaml(overwrite=False):
    config_file_name = 'config.yaml'
    if os.path.exists(config_file_name):
        if overwrite:
            write_config_file(config_file_name)
        else:
            print(f'{config_file_name} already exists.')
    else:
        write_config_file(config_file_name)


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    to_yaml(args.overwrite)
