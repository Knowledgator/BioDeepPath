import numpy as np
import random
from utils import *


class KGEnvironment(object):
    """knowledge graph environment definition"""

    def __init__(self, dataPath, task=None):
        f1 = open(dataPath + 'entities.tsv')
        f2 = open(dataPath + 'relations.tsv')
        self.entity2id = f1.readlines()
        self.relation2id = f2.readlines()
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations = []

        for line in self.entity2id:
            line = line.split()
            self.entity2id_[line[1]] = int(line[0])
        for line in self.relation2id:
            line = line.split()
            self.relation2id_[line[1]] = int(line[0])
            self.relations.append(line[1])
        self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
        self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')

        self.path = []
        self.path_relations = []

        # Knowledge Graph for path finding
        f = open(dataPath + 'kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()

        self.kb = []
        if task != None:
            relation = task.split()[2]
            source =  task.split()[0]
            target =  task.split()[1]
            for line in kb_all:
                r = line.split()[1]
                s = line.split()[0]
                t = line.split()[2]
                if s != source or t != target or r != relation or r!= relation + '_inv':
                    self.kb.append(line)


        self.die = 0  # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: (reward, [new_postion, target_position], done)
        '''
        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosed_relation = self.relations[action]
        print(curr_pos, chosed_relation, target_pos, len(self.kb))
        choices = []
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]]

            if curr_pos == e1_idx and triple[1] == chosed_relation and triple[2] in self.entity2id_:
                choices.append(triple)
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state  # stay in the initial state
            next_state[-1] = self.die
            return (reward, next_state, done)
        else:  # find a valid step
            path = random.choice(choices)
            self.path.append(path[1] + ' -> ' + path[2])
            self.path_relations.append(path[1])
            # print 'Find a valid step', path
            # print 'Action index', action
            self.die = 0
            new_pos = self.entity2id_[path[2]]
            reward = 0
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print('Find a path:', self.path)
                done = 1
                reward = 0
                new_state = None
            return (reward, new_state, done)

    def idx_state(self, idx_list):
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]
            targ = self.entity2vec[idx_list[2], :]
            return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
        else:
            return None

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[1]])
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))


