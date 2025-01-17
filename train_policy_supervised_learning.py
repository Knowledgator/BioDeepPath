from __future__ import division
import argparse

import numpy as np
from itertools import count
import os, sys

from networks import PolicyNetwork

from utils import *
from environment import KGEnvironment
from BFS.KB import KB
from BFS.BFS import BFS
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

all_args = argparse.ArgumentParser()
all_args.add_argument("-d", "--dataset", required=True,
   help="name of dataset to use")
all_args.add_argument("-t", "--task", required=False, default=True,
   help="relation name for training a model")
all_args.add_argument("-ed", "--emb_dim", required=False, default=500,
   help="dimensinality of initial embeddings of enities and relations")
all_args.add_argument("-em", "--emb_path", required=False, default='../ckpts/TransE_l2_BusinessLink_0',
   help="dimensinality of hiden state of policy network")
all_args.add_argument("-r", "--raw", required=False, default=1,
   help="does file with triples converted to")
all_args.add_argument("-m", "--model", required=False, default='TransE',
   help="model name")
args = vars(all_args.parse_args())

model_dir = 'models'
model_name = 'DeepPath_supervised_'

state_dim = args['emb_dim']*2
embedding_dim = args['emb_dim']
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50


def train_deep_path():

    policy_network = PolicyNetwork(state_dim, action_space).to(device)
    f = open(os.path.join('../Data', args['dataset'], 'train.tsv'))
    train_data = f.readlines()[:25]
    f.close()
    num_samples = len(train_data)

    if num_samples > 500:
        num_samples = 500
    else:
        num_episodes = num_samples

    # Knowledge Graph for path finding
    kids = Kids(args)
    kb = create_kb(args, kids)

    for episode in range(num_samples):
        print("Episode %d" % episode)
        print('Training Sample:', train_data[episode % num_samples][:-1])

        env = KGEnvironment(kb, kids, train_data[episode % num_samples])
        sample = train_data[episode % num_samples].split()
        # good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
        except Exception as e:
            print('Cannot find a path')
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            state_batch = torch.FloatTensor(state_batch).squeeze(dim=1).to(device)
            action_batch = torch.LongTensor(action_batch).to(device)
            prediction = policy_network(state_batch)
            loss = policy_network.compute_loss(prediction, action_batch)
            loss.backward()

            policy_network.optimizer.step()

    # save model
    print("Saving model to disk...")
    torch.save(policy_network.cpu(), os.path.join(model_dir, model_name + '.pt'))


def test(test_episodes):

    f = open(relationPath)
    test_data = f.readlines()
    f.close()
    test_num = len(test_data)

    test_data = test_data[-test_episodes:]
    print(len(test_data))
    success = 0

    policy_network = torch.load(os.path.join(model_dir, model_name + '.pt')).to(device)
    print('Model reloaded')
    # Knowledge Graph for path finding
    kb = create_kb(graphpath)
    kids = Kids(dataPath)

    for episode in range(len(test_data)):
        print('Test sample %d: %s' % (episode, test_data[episode][:-1]))
        env = KGEnvironment(kb, kids, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        for t in count():
            state_vec = torch.from_numpy(env.idx_state(state_idx)).float().to(device)
            action_probs = policy_network(state_vec)
            action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs.detach().numpy()))
            reward, new_state, done = env.interact(state_idx, action_chosen)
            if done or t == max_steps_test:
                if done:
                    print('Success')
                    success += 1
                print('Episode ends\n')
                break
            state_idx = new_state

    print('Success percentage:', success / test_episodes)

if __name__ == "__main__":
    train_deep_path()
