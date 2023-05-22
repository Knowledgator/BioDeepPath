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
from collections import OrderedDict
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import json


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


def from_pykeen_to_torchkge_dataset(
    identifier, split="training", max_num_examples=-1, *args, **kwargs
):
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
    if max_num_examples == -1:
        triples = splitted_dataset.triples
    elif max_num_examples > 0:
        triples = splitted_dataset.triples[:max_num_examples, :]
    else:
        raise ValueError(
            "Expected `max_num_examples` to be equal to -1 or "
            f"bigger than zero. Recieved: {max_num_examples}"
        )

    print("Creating DataFrame...")
    df = pd.DataFrame(
        {
            "from": triples[:, 0],
            "to": triples[:, 2],
            "rel": triples[:, 1],
        }
    )
    print("DataFrame created...")

    print("Creating Knowledge Graph...")
    kg = KnowledgeGraph(
        df=df,
        ent2ix=splitted_dataset.entity_to_id,
        rel2ix=splitted_dataset.relation_to_id,
    )
    print("Knowledge Graph created...")
    return kg


class KnowledgeGraphTokenizer:
    def __init__(self) -> None:
        self.entity_to_id = OrderedDict()
        self.id_to_entity = OrderedDict()
        self.relation_to_id = OrderedDict()
        self.id_to_relation = OrderedDict()

    def add_entity_to_tokenizer(self, entity):
        if entity not in self.entity_to_id:
            idx = len(self.entity_to_id)
            self.entity_to_id[entity] = idx
            self.id_to_entity[idx] = entity

    def add_relation_to_tokenizer(self, relation):
        if relation not in self.relation_to_id:
            idx = len(self.relation_to_id)
            self.relation_to_id[relation] = idx
            self.id_to_relation[idx] = relation

    def encode_entity(self, entity):
        return self.entity_to_id[entity]

    def encode_relation(self, relation):
        return self.relation_to_id[relation]

    def decode_entity(self, entity_id):
        return self.id_to_entity[entity_id]

    def decode_relation(self, relation_id):
        return self.id_to_relation[relation_id]

    def to_json(self):
        with open("entities_ids.json", "w") as f:
            json.dump(self.entity_to_id, f)

        with open("relations_ids.json", "w") as f:
            json.dump(self.relation_to_id, f)

    @classmethod
    def from_json(cls):
        instance = cls()

        with open("entities_ids.json", "r") as f:
            instance.entity_to_id = OrderedDict(
                json.load(
                    f, object_hook=lambda d: {k: int(v) for k, v in d.items()}
                )
            )
            instance.id_to_entity = {
                v: k for k, v in instance.entity_to_id.items()
            }

        with open("relations_ids.json", "r") as f:
            instance.relation_to_id = OrderedDict(
                json.load(
                    f, object_hook=lambda d: {k: int(v) for k, v in d.items()}
                )
            )
            instance.id_to_relation = {
                v: k for k, v in instance.relation_to_id.items()
            }

        return instance


def _from_txt_file_to_dataframe_and_tokenizer(
    filename: str,
    sep: str = "\t",
    order: List = ["from", "rel", "to"],
    header_row_exists=True,
    test_size=0.1,
) -> Tuple[pd.DataFrame, KnowledgeGraphTokenizer]:

    if len([o for o in order if o in {"from", "rel", "to"}]) < 3:
        raise ValueError(
            "Expected `order` to be a list that contains "
            f'`["from", "rel", "to"]`. Recieved: {order}'
        )

    tokenizer = KnowledgeGraphTokenizer()
    from_idx = order.index("from")
    rel_idx = order.index("rel")
    to_idx = order.index("to")
    contents = []

    with open(filename, "r") as f:
        if header_row_exists:
            _ = f.readline().strip().split(sep)

        for line in f:
            current_row = line.strip().lower().split(sep)

            if len(current_row) < 3:
                continue

            tokenizer.add_entity_to_tokenizer(current_row[from_idx])
            tokenizer.add_entity_to_tokenizer(current_row[to_idx])
            tokenizer.add_relation_to_tokenizer(current_row[rel_idx])
            contents.append(
                (
                    current_row[from_idx],
                    current_row[to_idx],
                    current_row[rel_idx],
                )
            )
        df = pd.DataFrame.from_dict(contents)

        df.columns = ["from", "to", "rel"]

        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=7, shuffle=True
        )

        df.to_csv("training_set.csv", index=False)
        df_test.to_csv("testing_set.csv", index=False)
        tokenizer.to_json()
        return df_train, tokenizer


def from_txt_to_dataset(
    filename: str,
    sep: str = "\t",
    order: List = ["from", "rel", "to"],
    header_row_exists=True,
):
    df, tokenizer = _from_txt_file_to_dataframe_and_tokenizer(
        filename, sep, order, header_row_exists
    )

    dataset = KnowledgeGraph(
        df,
        ent2ix=dict(tokenizer.entity_to_id),
        rel2ix=dict(tokenizer.relation_to_id),
    )
    return dataset
