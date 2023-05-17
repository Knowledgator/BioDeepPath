import torch
from torch.utils import data

from environment import Env


class SupervisedLearningDataset(data.Dataset):
    def __init__(self, kg_dataset: data.Dataset,
                 env: Env,
                 num_generated_episodes: int):
        super().__init__()
        self.kg_dataset = kg_dataset
        self.env = env
        self.num_generated_episodes = num_generated_episodes

    def __len__(self):
        return len(self.kg_dataset)

    def __getitem__(self, idx):
        h, t, _ = self.kg_dataset[idx]
        episodes = self.env.generate_episodes(h, t,
                                              self.num_generated_episodes)
        state_batch = []
        action_batch = []
        for episode in episodes:
            for transition in episode:
                state_batch.append(transition.state)
                action_batch.append(torch.tensor([transition.action]))
        state_batch = torch.cat(state_batch, dim=0)
        action_batch = torch.cat(action_batch, dim=0)
        return state_batch, action_batch
