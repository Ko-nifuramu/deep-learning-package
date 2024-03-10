import torch.nn as nn
import torch


class GRU_(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        z_dim: int,
        _activation=nn.ReLU(),
    ):
        super(GRU_, self).__init__()
        self.z_dim = z_dim
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_size, 288),
            _activation,
            nn.Linear(288, z_dim),
        )

        self.log_var_layer = nn.Sequential(
            nn.Linear(hidden_size, 288),
            _activation,
            nn.Linear(288, z_dim),
        )

        self.joint_layer = nn.Sequential(
            nn.Linear(hidden_size, 160),
            nn.BatchNorm1d(num_features=160),
            _activation,
            nn.Linear(160, 80),
            nn.BatchNorm1d(num_features=80),
            _activation,
            nn.Linear(80, 30),
            nn.Linear(30, joint_dim),
        )

    def forward(self, z_i_t, z_i_g, j_pre):

        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)
        # print(f'x : {x.shape}')
        x, _ = self.rnn(x, None)

        # print(f"x.shape : {x.shape}")
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], joint_dim)

        return mean, log_var, j_next_out

    def autoregress(self, z_i_t, z_i_g, j_pre, h):
        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)

        x, h = self.rnn(x, h)

        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], joint_dim)

        return mean, log_var, j_next_out, h


class RNN_(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        z_dim: int,
        _activation=nn.ReLU(),
    ):
        super(RNN_, self).__init__()
        self.z_dim = z_dim
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_size, 288),
            _activation,
            nn.Linear(288, z_dim),
        )

        self.log_var_layer = nn.Sequential(
            nn.Linear(hidden_size, 288),
            _activation,
            nn.Linear(288, z_dim),
        )

        self.joint_layer = nn.Sequential(
            nn.Linear(hidden_size, 160),
            nn.BatchNorm1d(num_features=160),
            _activation,
            nn.Linear(160, 80),
            nn.BatchNorm1d(num_features=80),
            _activation,
            nn.Linear(80, 30),
            nn.Linear(30, joint_dim),
        )

    def forward(self, z_i_t, z_i_g, j_pre):

        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)
        # print(f'x : {x.shape}')
        x, _ = self.rnn(x, None)

        # print(f"x.shape : {x.shape}")
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], joint_dim)

        return mean, log_var, j_next_out

    def autoregress(self, z_i_t, z_i_g, j_pre, h):
        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)

        x, h = self.rnn(x, h)

        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], joint_dim)

        return mean, log_var, j_next_out, h


class LSTM_(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        z_dim: int,
        _activation=nn.ReLU(),
    ):
        super(GRU_, self).__init__()
        self.z_dim = z_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_size, 288),
            _activation,
            nn.Linear(288, z_dim),
        )

        self.log_var_layer = nn.Sequential(
            nn.Linear(hidden_size, 288),
            _activation,
            nn.Linear(288, z_dim),
        )

        self.joint_layer = nn.Sequential(
            nn.Linear(hidden_size, 160),
            nn.BatchNorm1d(num_features=160),
            _activation,
            nn.Linear(160, 80),
            nn.BatchNorm1d(num_features=80),
            _activation,
            nn.Linear(80, 30),
            nn.Linear(30, joint_dim),
        )

    def forward(self, z_i_t, z_i_g, j_pre):

        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)
        # print(f'x : {x.shape}')
        x, _ = self.lstm(x, None)

        # print(f"x.shape : {x.shape}")
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], joint_dim)

        return mean, log_var, j_next_out

    def autoregress(self, z_i_t, z_i_g, j_pre, h):
        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)

        x, (c, h) = self.lstm(x, (c, h))

        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], joint_dim)

        return mean, log_var, j_next_out, c, h
