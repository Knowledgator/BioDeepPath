import os
import random
from itertools import count

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchkge import TransEModel
from tqdm.auto import tqdm

from environment import Env
from networks import PolicyNNV2, PolicyNNV3
from utils import Transition, construct_graph, from_pykeen_to_torchkge_dataset, from_txt_to_dataset
from transE_training import train_transE_model
from typing import Optional
from datasets import SupervisedLearningDataset

seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_space, learning_rate=0.0001):
        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.policy_nn = PolicyNNV2(state_dim, action_space)
        self.optimizer = optim.Adam(
            self.policy_nn.parameters(), lr=learning_rate
        )

    def forward(self, state):
        action_prob = self.policy_nn(state)
        return action_prob

    def compute_loss(self, action_prob, action, eps=1e-9):
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob + eps))
        return loss

    def compute_loss_rl(self, action_prob, target, action, eps=1e-9):
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob + eps) * target)
        return loss


def train_supervised(
    policy_model: PolicyNetwork,
    env: Env,
    train_ds: data.Dataset,
    num_epochs: int,
    num_generated_episodes: int,
    max_supervised_steps: int,
    device: str = "cuda",
    save_dir: Optional[str] = None,
):
    train_dl = data.DataLoader(train_ds, batch_size=1, shuffle=True)

    for i in range(1, num_epochs + 1):
        running_loss = 0
        with tqdm(train_dl, total=max_supervised_steps if max_supervised_steps != -1 else len(train_dl)) as iterator:
            for step, batch in enumerate(iterator):
                h, t, _ = int(batch[0][0]), int(batch[1][0]), int(batch[2][0])
                episodes = env.generate_episodes(h, t, num_generated_episodes)
                for episode in episodes:
                    state_batch = []
                    action_batch = []
                    for transition in episode:
                        state_batch.append(transition.state)
                        action_batch.append(torch.tensor([transition.action]))
                    state_batch = torch.cat(state_batch, dim=0).to(device)
                    action_batch = torch.cat(action_batch, dim=0).to(device)
                    policy_model.optimizer.zero_grad(set_to_none=True)
                    preds = policy_model(state_batch)
                    loss = policy_model.compute_loss(preds, action_batch)
                    loss.backward()
                    policy_model.optimizer.step()
                    running_loss += loss.item()
                iterator.set_description(
                    f"Epoch: {i}/{num_epochs} - "
                    f"Loss: {running_loss / len(train_dl)} - "
                    f"Found episodes: {len(episodes)}/{num_generated_episodes}"
                )

                if max_supervised_steps != -1 and step == max_supervised_steps:
                    break

        if save_dir is not None:
            weights_dir = os.path.join(
                save_dir, f"policy_sl_phase_weights_epoch_{i}.pt"
            )
            optimizer_dir = os.path.join(
                save_dir, f"policy_sl_phase_optimizer_epoch_{i}.pt"
            )
        else:
            weights_dir = f"policy_sl_phase_weights_epoch_{i}.pt"
            optimizer_dir = f"policy_sl_phase_optimizer_epoch_{i}.pt"

        torch.save(
            policy_model.policy_nn.state_dict(),
            weights_dir,
        )
        torch.save(
            policy_model.optimizer.state_dict(),
            optimizer_dir,
        )
    return policy_model


def train_rl(
    policy_model: PolicyNetwork,
    env: Env,
    train_ds: data.Dataset,
    num_episodes: int,
    max_steps: int,
    action_space: int,
    device: str = "cuda",
    save_dir: Optional[str] = None,
):
    done = False
    success = 0
    invalid_path = False
    policy_model = policy_model.train()
    sampled_indices = set()
    for episode in range(1, num_episodes + 1):
        state_batch_negative = []
        action_batch_negative = []
        episodes = []
        episode_path = ""

        sampled_idx = random.choice(range(len(train_ds)))
        while sampled_idx in sampled_indices:
            sampled_idx = random.choice(range(len(train_ds)))

        sampled_indices.add(sampled_idx)
        entity_1, entity_2 = train_ds[sampled_idx][:-1]
        current_state = env.reset_from(entity_1, entity_2)
        print(
            f"Episode: {episode} - Current Start: {env.current_head} - Current End: {env.current_target}"  # noqa
        )
        for step in count(1):
            episode_path += f"{env.current_head} -> "
            action_probs = policy_model(current_state.to(device))
            chosen_relation = np.random.choice(
                np.arange(action_space),
                p=np.squeeze(action_probs.cpu().detach().numpy()),
            )
            next_state, reward, done, invalid_path = env.step(chosen_relation)

            if reward == -1:
                state_batch_negative.append(current_state)
                action_batch_negative.append(chosen_relation)

            episodes.append(
                Transition(
                    state=current_state,
                    action=chosen_relation,
                    next_state=next_state,
                    reward=reward,
                )
            )
            if done or step == max_steps:
                episode_path += f"{env.current_target}"
                print("Episode Path:", episode_path)
                break

            current_state = next_state

        if len(state_batch_negative) != 0:
            print("Penalty to invalid steps:", len(state_batch_negative))
            state_batch_negative = torch.cat(state_batch_negative).to(device)
            action_batch_negative = torch.tensor(
                action_batch_negative, dtype=torch.long, device=device
            )
            policy_model.optimizer.zero_grad()
            predictions = policy_model(state_batch_negative)
            loss = policy_model.compute_loss_rl(
                predictions, -0.05, action_batch_negative
            )
            loss.backward()
            policy_model.optimizer.step()

        # If the agent success, do one optimization
        if not invalid_path:
            print("Success")

            success += 1
            path_length = len(env.episode_path)
            length_reward = 1 / path_length
            global_reward = 1
            total_reward = 0.1 * global_reward + 0.9 * length_reward
            state_batch = []
            action_batch = []
            for transition in episodes:
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.tensor(
                action_batch, device=device, dtype=torch.long
            )
            policy_model.optimizer.zero_grad()
            predictions = policy_model(state_batch)
            loss = policy_model.compute_loss_rl(
                predictions, total_reward, action_batch
            )
            loss.backward()
            policy_model.optimizer.step()
        else:
            global_reward = -0.05
            state_batch = []
            action_batch = []
            total_reward = global_reward
            for transition in episodes:
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            if len(state_batch) == 0:
                continue
            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.tensor(
                action_batch, device=device, dtype=torch.long
            )
            policy_model.optimizer.zero_grad()
            predictions = policy_model(state_batch)
            loss = policy_model.compute_loss_rl(
                predictions, total_reward, action_batch
            )
            loss.backward()
            policy_model.optimizer.step()

            print("Failed, Do one teacher guideline")
            good_episodes = env.generate_episodes(
                env.initial_head, env.current_target, 1
            )
            for item in good_episodes:
                teacher_state_batch = []
                teacher_action_batch = []
                total_reward = 0.7 * 1 + 0.3 * len(item)
                for transition in item:
                    teacher_state_batch.append(transition.state)
                    teacher_action_batch.append(transition.action)

                teacher_state_batch = (
                    torch.cat(teacher_state_batch)
                    .squeeze()
                    .to(device=device, dtype=torch.float32)
                )
                teacher_action_batch = torch.tensor(teacher_action_batch).to(
                    device=device, dtype=torch.long
                )
                policy_model.optimizer.zero_grad()
                predictions = policy_model(teacher_state_batch)
                loss = policy_model.compute_loss_rl(
                    predictions, 1, teacher_action_batch
                )
                loss.backward()
                policy_model.optimizer.step()

    if save_dir is not None:
        weights_dir = os.path.join(save_dir, "policy_rl_phase_weights.pt")
        optimizer_dir = os.path.join(save_dir, "policy_rl_phase_optimizer.pt")
    else:
        weights_dir = "policy_rl_phase_weights.pt"
        optimizer_dir = "policy_rl_phase_optimizer.pt"

    torch.save(policy_model.policy_nn.state_dict(), weights_dir)
    torch.save(policy_model.optimizer.state_dict(), optimizer_dir)
    return policy_model


def read_config_file(config_file_name):
    if not os.path.exists(config_file_name):
        raise FileNotFoundError(
            f"{config_file_name} does not exist. "
            "please call `create_config.py` first "
            "using `python create_config.py`."
        )

    with open(config_file_name, "r") as f:
        config = yaml.load(f, yaml.UnsafeLoader)
    return config


if __name__ == "__main__":
    args = read_config_file("config.yaml")

    print("Args:", args)

    if args.dataset_txt_file_path is not None:
        kg_train = from_txt_to_dataset(args.dataset_txt_file_path)
    elif args.kg_dataset is not None:
        kg_train = from_pykeen_to_torchkge_dataset(args.kg_dataset,
                                                   max_num_examples=args.max_num_examples)
    else:
        raise ValueError('`dataset_txt_file_path` and `kg_dataset` are None, '
                         'one of them should have a value.')

    if args.train_transE:
        print("Training TransE Model...")
        model = train_transE_model(
            kg_train,
            normalize_after_training=args.normalize_transE_weights,
            save_dir=args.save_weights_path,
            model_name=args.transE_weights_saved_name,
            epochs=args.transE_train_epochs,
        )
    else:
        model = TransEModel(
            emb_dim=args.transE_embed_dim,
            n_entities=kg_train.n_ent,
            n_relations=kg_train.n_rel,
        )
        model.load_state_dict(torch.load(os.path.join(args.save_weights_path,
                                                      args.transE_weights_saved_name)))
        print("TransE weights loaded.")

        if args.normalize_transE_weights:
            print('TransE weights normalized.')
            model.normalize_parameters()

    knowledge_graph = construct_graph(kg_train)
    env = Env(knowledge_graph, model)
    policy = PolicyNetwork(args.state_dim, kg_train.n_rel).to(args.device)
    if args.task == "supervised":
        train_supervised(
            policy_model=policy,
            env=env,
            train_ds=kg_train,
            num_epochs=args.num_supervised_epochs,
            num_generated_episodes=args.num_generated_episodes,
            max_supervised_steps=args.max_supervised_steps,
            device=args.device,
            save_dir=args.save_weights_path,
        )
    elif args.task == "rl":
        if args.rl_phase_load_from_checkpoint is not None:
            checkpoint = f"policy_sl_phase_weights_epoch_{args.rl_phase_load_from_checkpoint}"
            sl_saved_weights = os.path.join(args.save_weights_path, checkpoint)
            if os.path.exists(sl_saved_weights):
                policy.policy_nn.load_state_dict(torch.load(sl_saved_weights))

                print(f'Loaded from checkpoint {args.rl_phase_load_from_checkpoint}')

        train_rl(
            policy_model=policy,
            env=env,
            train_ds=kg_train,
            num_episodes=args.num_episods,
            max_steps=args.max_steps,
            action_space=kg_train.n_rel,
            device=args.device,
            save_dir=args.save_weights_path,
        )
    else:
        raise ValueError(
            "Unknown task, expected `supervised` or `rl` task. "
            f"Recieved: {args.task}."
        )
