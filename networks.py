import torch
import torch.nn as nn
import torch.nn.functional as F

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class MLP(nn.Module):
    def __init__(self, input_dim, n_epochs, batch_size):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, 1).to(device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.fc.parameters())
        self.loss_func = torch.nn.BCELoss()

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

    def train(self, training_features, train_labels):
        for epoch in range(self.n_epochs):
            order = torch.randperm(len(training_features))
            for start_index in range(
                0, len(training_features), self.batch_size
            ):
                self.optimizer.zero_grad()
                # fmt: off
                batch_indexes = order[
                    start_index: start_index + self.batch_size
                ]
                # fmt: on
                X_batch = training_features[batch_indexes].to(device)
                y_batch = train_labels[batch_indexes].to(device)

                preds = torch.sigmoid(self.fc(X_batch))

                loss_value = self.loss_func(preds.squeeze(), y_batch)
                loss_value.backward()

                self.optimizer.step()


class PolicyNNV2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNNV2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512, bias=True)
        self.layernorm_1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.layernorm_2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, action_dim, bias=True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, state):
        y = self.fc1(state)
        y = self.layernorm_1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.layernorm_2(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc3(y)
        action_probs = F.softmax(y, dim=-1)
        return action_probs


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.25) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        return self.dropout(F.gelu(self.layernorm(self.fc(inputs))))


class OutputBlock(nn.Module):
    def __init__(self, input_dim, action_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(1024, action_dim, bias=True)
        self.activation = nn.Softmax(dim=-1)


    def forward(self, inputs):
        return self.activation(self.fc(inputs))



class PolicyNNV3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNNV3, self).__init__()
        self.block_1 = Block(state_dim, 512)
        self.residual_fc = nn.Linear(state_dim, 1024)
        self.block_2 = Block(512, 1024)
        self.block_3 = Block(1024, 1024)
        self.output_block = OutputBlock(1024, action_dim)

    def forward(self, state):
        y = self.block_1(state)
        residual = self.residual_fc(state)
        y = self.block_2(y)
        y = self.block_3(y)
        y += residual

        action_probs = self.output_block(y)
        return action_probs


class PolicyNN(nn.Module):
    def __init__(self, state_dim, action_dim, initializer=None):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512, bias=True)
        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.fc3 = nn.Linear(1024, action_dim, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = self.dropout(y)
        y = F.relu(self.fc2(y))
        y = self.dropout(y)
        y = F.relu(self.fc3(y))
        action_probs = self.softmax(y)
        return action_probs


class ValueNN(nn.Module):
    def __init__(self, state_dim, initializer=None):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64, bias=True)
        self.fc2 = nn.Linear(64, 1, bias=True)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        value_estimated = self.fc2(y)
        return torch.squeeze(value_estimated)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        action_values = self.fc3(y)
        return action_values
