from __future__ import division
import argparse
import numpy as np
import collections
from itertools import count
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys, os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import PolicyNN, ValueNN
from utils import *
from environment import KGEnvironment
from networks import PolicyNetwork

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

all_args = argparse.ArgumentParser()
all_args.add_argument("-d", "--dataset", required=True,
   help="name of dataset to use")
all_args.add_argument("-t", "--task", required=False, default=True,
   help="relation name for training a model")
all_args.add_argument("-s", "--use_supervised", required=False, default = 0,
   help="0 or 1 if to use pre-trained supervised model")
all_args.add_argument("-ed", "--emb_dim", required=False, default=500,
   help="dimensinality of initial embeddings of enities and relations")
all_args.add_argument("-em", "--emb_path", required=False, default='../ckpts/TransE_l2_BusinessLink_0',
   help="dimensinality of hiden state of policy network")
all_args.add_argument("-r", "--raw", required=False, default=1,
   help="does file with triples converted to")
all_args.add_argument("-m", "--model", required=False, default='TransE',
   help="model name")
args = vars(all_args.parse_args())


with open(os.path.join('../Data', args['dataset'], 'relations.tsv')) as f:
    relation2id = {r.split()[1]:int(r.split()[0]) for r in f.read().split('\n')[:-1]}
    #action space - number of relations and reversed relations
    action_space = len(relation2id)

model_dir = 'models'
model_name = 'DeepPath_'

state_dim = args['emb_dim']*2
upload_model = args['use_supervised']
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

def get_training_pairs(args, kids,target_relation):
    if args['raw']:
        rel = target_relation
    else:
        rel = relation2id[target_relation]
    training_pairs = []
    with open(os.path.join('../Data', args['dataset'], 'train.tsv')) as f:
        train_triples_raw = f.read().split('\n')
        for ttw in train_triples_raw[:-1]:
            triple = ttw.split('\t')
            if triple[1] == rel:
                try:
                    s,r,t = triple
                except:
                    print(triple)
                    continue
                if not args['raw']:
                    s = kids.id2entity[int(s)]
                    r = kids.id2relation[int(r)]
                    t = kids.id2entity[int(t)]
                training_pairs.append([s,r,t])
    return training_pairs

def REINFORCE(policy_network, target_relation):
    # Knowledge Graph for path finding
    kids = Kids(args)
    kb = create_kb(args, kids)

    training_pairs = get_training_pairs(args, kids,target_relation)
    num_episodes = len(training_pairs)
    if num_episodes > 1000:
        num_episodes = 1000
    success = 0
    done = 0

    # path_found = set()
    path_found_entity = []
    path_relation_found = []

    for i_episode in range(num_episodes):
        start = time.time()
        sample = training_pairs[i_episode]
        state_idx = [kids.entity2id_[sample[0]], kids.entity2id_[sample[2]], 0]
        print('Episode %d' % i_episode)
        print('Training sample: ',sample)
        env = KGEnvironment(kb, kids, sample)

        episode = []
        state_batch_negative = []
        action_batch_negative = []
        for t in count():
            policy_network.eval()
            state_vec = torch.from_numpy(env.idx_state(state_idx)).float().to(device)
            with torch.no_grad():
                action_probs = policy_network(state_vec)
            try:
                action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs.cpu().detach().numpy()))
            except:
                continue
            # print(env. get_valid_actions(state_idx[0]))
            reward, new_state, done = env.interact(args, state_idx, action_chosen)

            if reward == -1:  # the action fails for this step
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)

            new_state_vec = env.idx_state(new_state)
            episode.append(Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

            if done or t == max_steps:
                break

            state_idx = new_state

        # Discourage the agent when it choose an invalid step
        if len(state_batch_negative) != 0:
            print('Penalty to invalid steps:', len(state_batch_negative))
            policy_network.train()
            policy_network.optimizer.zero_grad()
            state_batch_negative = torch.cat(state_batch_negative).to(device)
            action_batch_negative = torch.LongTensor(action_batch_negative).to(device)
            predictions = policy_network(state_batch_negative)
            loss = policy_network.compute_loss_rl(predictions, -0.1, action_batch_negative)
            loss.backward()
            policy_network.optimizer.step()

        print('----- FINAL PATH -----')
        print('\t'.join(env.path))
        print('PATH LENGTH', len(env.path))
        print('----- FINAL PATH -----')

        # If the agent success, do one optimization
        if done == 1:
            print('Success')

            path_found_entity.append(path_clean(' -> '.join(env.path)))

            success += 1
            path_length = len(env.path)
            length_reward = 1 #/ np.log2(path_length)
            global_reward = 1

            # if len(path_found) != 0:
            # 	path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_found]
            # 	curr_path_embedding = env.path_embedding(env.path_relations)
            # 	path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
            # 	cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
            # 	diverse_reward = -np.mean(cos_sim)
            # 	print 'diverse_reward', diverse_reward
            # 	total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
            # else:
            # 	total_reward = 0.1*global_reward + 0.9*length_reward
            # path_found.add(' -> '.join(env.path_relations))

            total_reward = 0.1 * global_reward + 0.9 * length_reward
            state_batch = []
            action_batch = []
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            policy_network.train()
            policy_network.optimizer.zero_grad()
            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.LongTensor(action_batch).to(device)
            predictions = policy_network(state_batch)
            loss = policy_network.compute_loss_rl(predictions, total_reward, action_batch)
            loss.backward()
            policy_network.optimizer.step()
        else:
            global_reward = -0.1
            # length_reward = 1/len(env.path)

            state_batch = []
            action_batch = []
            total_reward = global_reward
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            if len(state_batch) == 0:
                continue
            policy_network.train()
            policy_network.optimizer.zero_grad()
            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.LongTensor(action_batch).to(device)
            predictions = policy_network(state_batch)
            loss = policy_network.compute_loss_rl(predictions, total_reward, action_batch)
            loss.backward()
            policy_network.optimizer.step()

            # print('Failed, Do one teacher guideline')
            # try:      
            #     good_episodes = teacher(sample[0], sample[1], 1, env, graphpath)
            #     for item in good_episodes:
            #         teacher_state_batch = []
            #         teacher_action_batch = []
            #         total_reward = 0.0 * 1 + 1 * 1 / len(item)
            #         for t, transition in enumerate(item):
            #             teacher_state_batch.append(transition.state)
            #             teacher_action_batch.append(transition.action)
            #         teacher_state_batch = torch.FloatTensor(teacher_state_batch).squeeze().to(device)
            #         teacher_action_batch = torch.LongTensor(teacher_action_batch).to(device)
            #         predictions = policy_network(teacher_state_batch)
            #         loss = policy_network.compute_loss_rl(predictions, 1, teacher_action_batch)
            #         loss.backward()
            #         policy_network.optimizer.step()
            # except Exception as e:
            #     print('Teacher guideline failed')

        print('Episode time: ', time.time() - start)
        print('\n')
    print('Success percentage:', success / num_episodes)

    for path in path_found_entity:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)
    spath = os.path.join('../Data', args['dataset'], 'tasks', target_relation)
    if not os.path.isdir(spath):
        os.mkdir(spath)
    f = open(os.path.join('../Data', args['dataset'], 'tasks', target_relation, 'path_stats.txt'), 'w')
    for item in relation_path_stats:
        f.write(item[0] + '\t' + str(item[1]) + '\n')
    f.close()
    print('Path stats saved')

    return


def retrain(target_relation):
    epochs = 10
    # TODO: Fix this - load saved model and optimizer state to Policy_network.policy_nn.
    print('Start retraining')

    if upload_model:
        policy_network = torch.load(os.path.join(model_dir, 'policy_supervised_' + target_relation + '.pt'))
        print("sl_policy restored")
    else:
        policy_network = PolicyNetwork(state_dim, action_space).to(device)
        print("policy network was created")

    # for epoch in range(epochs):
    REINFORCE(policy_network, target_relation)
    # save model
    print("Saving model to disk...")
    torch.save(policy_network, os.path.join(model_dir, model_name + target_relation + '.pt'))
    print('Retrained model saved')

def test(target_relation):
    # Knowledge Graph for path finding
    kids = Kids(args)
    kb = create_kb(args, kids)
    training_pairs = get_training_pairs(args, kids, target_relation)

    test_data = training_pairs
    test_num = len(test_data)

    success = 0
    done = 0

    path_found = []
    path_relation_found = []
    path_set = set()


    policy_network = torch.load(os.path.join(model_dir, model_name + target_relation + '.pt')).to(device)
    print('Model reloaded')

    if test_num > 500:
        test_num = 500


    for episode in range(test_num):
        print('Test sample %d: %s' % (episode, test_data[episode][:-1]))
        env = KGEnvironment(kb, kids, test_data[episode])
        sample = test_data[episode]

        state_idx = [kids.entity2id_[sample[0]], kids.entity2id_[sample[2]], 0]

        transitions = []

        for t in count():
            state_vec = torch.from_numpy(env.idx_state(state_idx)).float().to(device)
            action_probs = policy_network(state_vec)
            try:
                action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs.cpu().detach().numpy()))
            except:
                continue
            reward, new_state, done = env.interact(args, state_idx, action_chosen)
            new_state_vec = env.idx_state(new_state)
            transitions.append(Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

            if done or t == max_steps_test:
                if done:
                    success += 1
                    print("Success\n")
                    path = path_clean(' -> '.join(env.path))
                    path_found.append(path)
                else:
                    print('Episode ends due to step limit\n')
                break
            state_idx = new_state

    for path in path_found:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    # path_stats = collections.Counter(path_found).items()
    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)

    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(' -> '))
        ranking_path.append((path, length))

    ranking_path = sorted(ranking_path, key=lambda x: x[1])
    print('Success percentage:', success / test_num)

    f = open(os.path.join('../Data', args['dataset'], 'tasks', target_relation, 'path_to_use.txt'), 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print('path to use saved')
    return


if __name__ == "__main__":
    if args['task']:
        print("Relation:", args['task'])
        retrain(args['task'])
        test(args['task'])   
    else:
        for relation in relation2id.keys():
            if not re.search('_rev', relation) and len(relation)!=1:
                print("Relation:", relation)
                retrain(relation)
                test(relation)
# retrain()
