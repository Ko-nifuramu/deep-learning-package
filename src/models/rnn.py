import torch
import torch.nn as nn


# jointを出力するモデルをactionモデルとして分けるか迷ったが、助長な気がしたのでRNNモデルに統合
class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rnn_hidden_dim: int,
        num_layers: int,
        joint_dim: int,
        z_dim: int,
        mean_hidden_dims: tuple = [288],
        log_var_hidden_dims: tuple = [288],
        joint_hidden_dims: tuple = [169, 80, 30],
        _activation=nn.ReLU(),
    ):
        super(GRU, self).__init__()
        self.z_dim = z_dim
        self.joint_dim = joint_dim
        self.rnn = nn.GRU(
            input_dim=input_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_layer = self._build_linear_layer(
            rnn_hidden_dim, mean_hidden_dims, z_dim
        )
        self.log_var_layer = self._build_linear_layer(
            rnn_hidden_dim, log_var_hidden_dims, z_dim
        )
        self.joint_layer = self._build_linear_layer(
            rnn_hidden_dim, joint_hidden_dims, joint_dim
        )

    def forward(self, z_i_t, z_i_g, j_pre):
        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, self.joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)

        x, _ = self.rnn(x, None)

        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], self.joint_dim)

        return mean, log_var, j_next_out

    def autoregress(self, z_i_t, z_i_g, j_pre, h_pre):
        z_i_t = z_i_t.reshape(j_pre.shape[0], -1, self.z_dim)
        z_i_g = z_i_g.reshape(j_pre.shape[0], -1, self.z_dim)
        j_pre = j_pre.reshape(j_pre.shape[0], -1, self.joint_dim)
        x = torch.concat([z_i_t, z_i_g, j_pre], dim=2)

        x, h_next = self.rnn(x, h_pre)

        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        x = x.reshape(-1, x.shape[-1])
        j_next_out = self.joint_layer(x)
        j_next_out = j_next_out.reshape(mean.shape[0], mean.shape[1], self.joint_dim)

        return mean, log_var, j_next_out, h_next

    def _build_linear_layer(self, input_dim: int, hidden_dims: tuple, output_dim: int):
        linear_layers = []
        for hidden_dim in hidden_dims:
            linear_layers.append(nn.Linear(input_dim, hidden_dim))
            linear_layers.append(self._activation)
            input_dim = hidden_dim

        linear_layers.append(nn.Linear(input_dim, output_dim))

        return nn.Sequential(*linear_layers)
