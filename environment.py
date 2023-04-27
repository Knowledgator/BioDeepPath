import random
from collections import deque
from typing import List

import networkx as nx
import torch
from torch.utils.data import Dataset

from utils import Transition


class Env:
    def __init__(
        self,
        graph: nx.DiGraph,
        dataset: Dataset,
        trans_e_model: torch.nn.Module,
        relations: List,
    ):
        self.graph = graph
        self.dataset = dataset
        self.nodes = list(graph.nodes)
        self.trans_e_model = trans_e_model
        self.relations = relations
        self.current_head, self.current_tail = None, None
        self.path = []
        self.trans_e_model.to("cpu").eval()
        self.entities = [i[0] for i in dataset]

    def dfs(self, start, end):
        """Performs DFS and path tracking."""
        visited = set()
        path = []
        stop_dfs = False

        def _dfs(v):
            """Nested function to perform DFS."""
            nonlocal self, end, visited, stop_dfs, path
            path.append(v)
            if end in self.graph[v]:
                print("FOUND")
                path.append(end)
                stop_dfs = True
                return path
            visited.add(v)
            for x in self.graph[v]:
                if x not in visited and not stop_dfs and x in self.graph:
                    return _dfs(x)

        return _dfs(start) or []

    def bfs(self, start, end, verbose=False):
        """Performs BFS and path construction."""
        # key: child node, value: parent node
        # To keep track of each node and its parent
        # to be used later to construct
        # the path from the start to target entities.
        path = {start: None}
        # Stores the path from start to target entities.
        constructed_path = [end]

        def _bfs():
            """Nested function for performing BFS."""
            nonlocal self, start, end, path, verbose
            visited = set()

            # BFS queue
            queue = deque([start])

            while len(queue) > 0:
                current_node = queue.popleft()
                visited.add(current_node)
                for node in self.graph[current_node]:
                    if node not in visited and node in self.graph:
                        queue.append(node)
                        path[node] = current_node
                    if node == end:
                        if verbose:
                            print("FOUND")
                        return
                    visited.add(node)

        def _construct_path():
            """Nested function to construct the path."""
            nonlocal start, end, constructed_path, path
            while constructed_path[-1] != start:
                constructed_path.append(path[constructed_path[-1]])
            constructed_path.reverse()

        _bfs()
        _construct_path()
        return constructed_path

    def sample_path(self):
        """Samples random start and end entities and finds the path between them."""
        start_node = random.choice(self.nodes)
        end_node = random.choice(list(self.graph.nodes - [start_node]))

        print(f"start node: {start_node} - end node: {end_node}")
        path = self.bfs(start_node, end_node)
        if len(path) == 0:
            print(f"No path found from {start_node} to {end_node}. Trying again...")
            return self.sample_path()
        return path

    def get_state_embedding(self, state):
        """Returns the embedding of the state."""
        current_entity, target_entity = state
        with torch.no_grad():
            current_entity_embed = self.trans_e_model.ent_emb(
                torch.tensor([current_entity])
            )
            target_entity_embed = self.trans_e_model.ent_emb(
                torch.tensor([target_entity])
            )
        new_state = (current_entity_embed, target_entity_embed - current_entity_embed)
        new_state = torch.concatenate(new_state, dim=-1)
        return new_state

    def pick_random_intermediates_between(self, entity1, entity2, num_paths):
        """Generate intermediate paths between two entities."""
        intermediate_nodes = set()
        if num_paths > len(self.entities) - 2:
            raise ValueError(
                "Number of Intermediates picked is larger than possible",
                "num_entities: {}".format(len(self.entities)),
                "num_itermediates: {}".format(num),
            )
        for i in range(num_paths):
            itermediate = random.choice(self.entities)
            while (
                itermediate in intermediate_nodes
                or itermediate == entity1
                or itermediate == entity2
            ):
                itermediate = random.choice(self.entities)
            intermediate_nodes.add(itermediate)
        return list(intermediate_nodes)

    def sample_two_entities(self, verbose=False):
        """Samples random start and end entities
        and ensures that they have paths.
        """
        start_node = random.choice(self.nodes)
        end_node = random.choice(list(self.graph.nodes - [start_node]))

        if verbose:
            print(f"start node: {start_node} - end node: {end_node}")
        try:
            path = self.bfs(start_node, end_node)
            if len(path) == 0:
                print(f"No path found from {start_node} to {end_node}. Trying again...")
                return self.sample_two_entities()
            else:
                return (start_node, end_node)
        except Exception:
            print(f"No path found from {start_node} to {end_node}. Trying again...")
            return self.sample_two_entities()

    def step(self, action):
        """Takes an action and updates the
        environment by returning a reward and a new state.
        """
        tails = self.dataset.dict_of_tails[(self.current_head, action)]
        reward = 0
        done = False
        invalid_path = False

        if self.current_target in tails:
            done = True
            reward = 0
            new_state = None

        if not done:
            possible_paths = []
            for tail in tails:
                try:
                    path = self.bfs(tail, self.current_target)
                except Exception:
                    continue
                possible_paths.append((path, tail))

            if len(possible_paths) == 0:
                new_state = None
                reward = -1
                done = True
                invalid_path = True
            else:
                chosen_path = random.choice(possible_paths)
                self.current_head = chosen_path[1]
                self.path.append(self.current_head)
                new_state = self.get_state_embedding(
                    (self.current_head, self.current_target)
                )

        return new_state, reward, done, invalid_path

    def generate_episodes(self, entity_1, entity_2, num_paths, verbose=False):
        """Generates episodes by generating paths between two entities."""
        intermediate_paths = self.pick_random_intermediates_between(
            entity_1, entity_2, num_paths
        )
        paths = []
        relations = []
        for intermediate_path in intermediate_paths:
            try:
                entity_1_to_current_path = self.bfs(
                    entity_1, intermediate_path, verbose
                )
                entity_2_to_current_path = self.bfs(
                    intermediate_path, entity_2, verbose
                )
                entity_1_to_current_path.extend(entity_2_to_current_path[1:])
                paths.append(entity_1_to_current_path)
                relations.append(
                    [
                        self.graph[entity_1_to_current_path[idx]][
                            entity_1_to_current_path[idx + 1]
                        ]["relation"]
                        for idx in range(len(entity_1_to_current_path) - 1)
                    ]
                )
            except Exception:
                print(
                    f"Could not find a path at intermediate point: {intermediate_path}. Will be skipped."
                )
        good_episodes = []
        target_id = entity_2
        for path, relation in zip(paths, relations):
            good_episode = []
            for i in range(len(path) - 1):
                curr_id = path[i]
                next_id = path[i + 1]
                state_curr = (curr_id, target_id)
                state_next = (next_id, target_id)
                action_id = relation[i]
                good_episode.append(
                    Transition(
                        state=self.get_state_embedding(state_curr),
                        action=action_id,
                        next_state=self.get_state_embedding(state_next),
                        reward=1,
                    )
                )
            good_episodes.append(good_episode)
        return good_episodes

    def reset(self, verbose=False):
        """Returns initial state."""
        sampled_path = self.sample_two_entities(verbose=verbose)
        self.current_head, self.current_target = sampled_path
        self.initial_head = self.current_head
        self.path = []
        self.path.append(self.current_head)
        initial_state = self.get_state_embedding(sampled_path)
        return initial_state
