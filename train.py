from argparse import ArgumentParser
from itertools import count

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchkge import TransEModel
from torchkge.utils import load_fb15k
from tqdm.auto import tqdm

from environment import Env
from networks import PolicyNNV2
from utils import Transition, construct_graph


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

    def compute_loss(self, action_prob, action):
        # TODO: Add regularization loss
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob))
        return loss

    def compute_loss_rl(self, action_prob, target, action):
        # TODO: Add regularization loss
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob) * target)
        return loss


def train_supervised(policy_model, env, train_ds, num_epochs, device="cuda"):
    for i in range(1, num_epochs + 1):
        running_loss = 0

        train_dl = data.DataLoader(train_ds, batch_size=1, shuffle=True)
        with tqdm(train_dl) as iterator:
            for batch in iterator:
                h, t, _ = int(batch[0][0]), int(batch[1][0]), int(batch[2][0])
                episodes = env.generate_episodes(h, t, 5)
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
                    f"Epoch: {i}/{num_epochs} - Loss: {running_loss / len(train_dl)}"
                )

        torch.save(
            policy_model.state_dict(), f"policy_model_weights_supervised_{i}.pt"
        )
        torch.save(
            policy_model.optimizer.state_dict(),
            f"policy_optimizer_supervised_{i}.pt",
        )
    return policy_model


def train_rl(
    policy_model, env, num_episodes, max_steps, action_space, device="cuda"
):
    done = False
    success = 0
    invalid_path = False
    policy_model = policy_model.train()
    for episode in range(1, num_episodes + 1):
        state_batch_negative = []
        action_batch_negative = []
        episodes = []
        episode_path = ""

        current_state = env.reset()
        print(
            f"Episode: {episode} - Current Start: {env.current_head} - Current End: {env.current_target}"
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
            path_length = len(env.path)
            length_reward = 1 / path_length
            global_reward = 1
            total_reward = 0.1 * global_reward + 0.9 * length_reward
            state_batch = []
            action_batch = []
            for t, transition in enumerate(episodes):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.tensor(
                action_batch, device=device, dtype=torch.long
            )
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
            for t, transition in enumerate(episodes):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            if len(state_batch) == 0:
                continue
            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.tensor(
                action_batch, device=device, dtype=torch.long
            )
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
                for t, transition in enumerate(item):
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
                predictions = policy_model(teacher_state_batch)
                loss = policy_model.compute_loss_rl(
                    predictions, total_reward, teacher_action_batch
                )
                loss.backward()
                policy_model.optimizer.step()

    torch.save(policy_model.state_dict(), "policy_rl_weights.pt")
    torch.save(policy_model.optimizer.state_dict(), "policy_optimizer_rl.pt")
    return policy_model


def create_arg_parser():
    argparser = ArgumentParser()
    argparser.add_argument("--transE_embed_dim", default=100, type=int)
    argparser.add_argument("--state_dim", required=False, default=200, type=int)
    argparser.add_argument(
        "--action_space", required=False, default=1345, type=int
    )
    argparser.add_argument("--eps_start", required=False, default=1, type=int)
    argparser.add_argument("--eps_end", required=False, default=0.1, type=float)
    argparser.add_argument(
        "--epe_decay", required=False, default=1000, type=int
    )
    argparser.add_argument(
        "--replay_memory_size", required=False, default=10000, type=int
    )
    argparser.add_argument(
        "--batch_size", required=False, default=128, type=int
    )
    argparser.add_argument(
        "--embedding_dim", required=False, default=100, type=int
    )
    argparser.add_argument("--gamma", required=False, default=0.99, type=float)
    argparser.add_argument(
        "--target_update_freq", required=False, default=1000, type=int
    )
    argparser.add_argument("--max_steps", required=False, default=50, type=int)
    argparser.add_argument(
        "--max_steps_test", required=False, default=50, type=int
    )
    argparser.add_argument(
        "--num_episods", required=False, default=2000, type=int
    )
    argparser.add_argument(
        "--device", required=False, default="cuda:0", type=str
    )
    return argparser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    kg_train, _, _ = load_fb15k()
    model = TransEModel(
        args.transE_embed_dim,
        kg_train.n_ent,
        kg_train.n_rel,
        dissimilarity_type="L2",
    )
    G = construct_graph(kg_train)
    env = Env(G, kg_train, model, kg_train.relations.tolist())
    policy = PolicyNetwork(args.state_dim, args.action_space).to(args.device)
    train_supervised(policy, env, kg_train, 2)
    train_rl(policy, env, kg_train, args.num_episods, args.max_steps)
