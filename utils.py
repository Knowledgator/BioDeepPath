from __future__ import division

from collections import Counter, namedtuple

import networkx as nx
import numpy as np

from BFS.BFS import BFS
from BFS.KB import KB

try:
    from pykeen import datasets as pykeen_datasets
except ImportError:
    pykeen_datasets = None

from torchkge.data_structures import KnowledgeGraph
import pandas as pd


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class Kids:
    def __init__(self, dataPath):
        f1 = open(dataPath + "entities.tsv")
        f2 = open(dataPath + "relations.tsv")
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
        self.id2entity = {v: k for k, v in self.entity2id_.items()}
        self.id2relation = {v: k for k, v in self.relation2id_.items()}
        self.entity2vec = np.loadtxt(dataPath + "entity2vec.bern")
        self.relation2vec = np.loadtxt(dataPath + "relation2vec.bern")


def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
    return sum(v1 == v2)


def create_kb(graphpath):
    f = open(graphpath)
    kb_all = f.readlines()
    f.close()
    kb = {}
    for line in kb_all:
        r = line.split()[1]
        s = line.split()[0]
        t = line.split()[2]
        if s not in kb:
            kb[s] = {r: {t}}
        elif r not in kb[s]:
            kb[s][r] = {t}
        else:
            kb[s][r].add(t)
    return kb


def teacher(e1, e2, num_paths, env, path=None):
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)
    kb.removePath(e1, e2)
    intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
    res_entity_lists = []
    res_path_lists = []
    for i in range(num_paths):
        suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
        suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
        if suc1 and suc2:
            res_entity_lists.append(entity_list1 + entity_list2[1:])
            res_path_lists.append(path_list1 + path_list2)
    print("BFS found paths:", len(res_path_lists))

    # ---------- clean the path --------
    res_entity_lists_new = []
    res_path_lists_new = []
    for entities, relations in zip(res_entity_lists, res_path_lists):
        rel_ents = []
        for i in range(len(entities) + len(relations)):
            if i % 2 == 0:
                rel_ents.append(entities[int(i / 2)])
            else:
                rel_ents.append(relations[int(i / 2)])

        # print rel_ents

        entity_stats = Counter(entities).items()
        duplicate_ents = [item for item in entity_stats if item[1] != 1]
        duplicate_ents.sort(key=lambda x: x[1], reverse=True)
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
            if len(ent_idx) != 0:
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx != max_idx:
                    rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
        entities_new = []
        relations_new = []
        for idx, item in enumerate(rel_ents):
            if idx % 2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item)
        res_entity_lists_new.append(entities_new)
        res_path_lists_new.append(relations_new)

    print(res_entity_lists_new)
    print(res_path_lists_new)

    good_episodes = []
    targetID = env.kids.entity2id_[e2]
    for path in zip(res_entity_lists_new, res_path_lists_new):
        good_episode = []
        for i in range(len(path[0]) - 1):
            currID = env.kids.entity2id_[path[0][i]]
            nextID = env.kids.entity2id_[path[0][i + 1]]
            state_curr = [currID, targetID, 0]
            state_next = [nextID, targetID, 0]
            actionID = env.kids.relation2id_[path[1][i]]
            good_episode.append(
                Transition(
                    state=env.idx_state(state_curr),
                    action=actionID,
                    next_state=env.idx_state(state_next),
                    reward=1,
                )
            )
        good_episodes.append(good_episode)
    return good_episodes


def path_clean(path):
    rel_ents = path.split(" -> ")
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return " -> ".join(rel_ents)


def prob_norm(probs):
    return probs / sum(probs)


def get_relationships(ds):
    relations = []
    for rels in ds.dict_of_rels.values():
        for rel in rels:
            if rel not in relations:
                relations.append(rel)

    return relations


def construct_graph(ds):
    G = nx.DiGraph()
    attrs = {}
    for h, t, r in ds:
        G.add_edge(h, t)
        attrs[(h, t)] = r
    nx.set_edge_attributes(G, attrs, "relation")
    return G


def pykeen_to_torchkge_dataset(identifier, split="training", *args, **kwargs):
    if pykeen_datasets is None:
        raise ImportError(
            "In order to use a dataset from `pykeen` you need to "
            "install `pykeen` using `pip install pykeen`."
        )
    dataset = getattr(pykeen_datasets, identifier, None)
    if dataset is None:
        raise ValueError(f"Dataset not found. Recieved: {identifier}.")

    dataset = dataset(*args, **kwargs)
    splitted_dataset = getattr(dataset, split)
    triples = splitted_dataset.triples
    df = pd.DataFrame(
        {
            "from": triples[:, 0],
            "to": triples[:, 2],
            "rel": triples[:, 1],
        }
    )
    kg = KnowledgeGraph(
        df=df,
        ent2ix=splitted_dataset.entity_to_id,
        rel2ix=splitted_dataset.relation_to_id,
    )
    return kg
