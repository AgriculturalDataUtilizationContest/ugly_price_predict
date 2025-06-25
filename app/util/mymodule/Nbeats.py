from typing import Tuple

import numpy as np
import torch as t


class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size,
        theta_size: int,
        basis_function: t.nn.Module,
        layers: int,
        layer_size: int,
    ):
        """
        N-BEATS block.

        :param input_size: Input size (length of input time series for each time step).
        :param theta_size: Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers in the block (hidden layers).
        :param layer_size: Number of units in each hidden layer.
        """
        super().__init__()
        self.layers = t.nn.ModuleList(
            [t.nn.Linear(in_features=input_size, out_features=layer_size)]
            + [
                t.nn.Linear(in_features=layer_size, out_features=layer_size)
                for _ in range(layers - 1)
            ]
        )
        self.basis_parameters = t.nn.Linear(
            in_features=layer_size, out_features=theta_size
        )
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """
        Forward pass for the N-BEATS block.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: backcast and forecast from the basis function (tuple of tensors)
                 with shape (batch_size, backcast_size), (batch_size, forecast_size)
        """
        block_input = x  # (batch_size, input_size)
        for layer in self.layers:
            block_input = t.relu(
                layer(block_input)
            )  # After each layer, shape remains (batch_size, layer_size)

        basis_parameters = self.basis_parameters(
            block_input
        )  # (batch_size, theta_size)
        return self.basis_function(basis_parameters)  # (backcast, forecast)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        """
        Forward pass for the N-BEATS model.

        :param x: Input tensor of shape (batch_size, time_steps)
        :param input_mask: Mask tensor of shape (batch_size, time_steps)
        :return: final forecast tensor of shape (batch_size, forecast_size)
        """
        residuals = x.flip(
            dims=(1,)
        )  # Flip input to reverse time direction (shape: (batch_size, time_steps))
        input_mask = input_mask.flip(
            dims=(1,)
        )  # Flip the mask similarly (shape: (batch_size, time_steps))

        forecast = x[
            :, -1:
        ]  # Forecast starts as the last element in x (shape: (batch_size, 1))

        # Process through all blocks
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                residuals
            )  # (backcast: batch_size, backcast_size), (forecast: batch_size, forecast_size)
            residuals = (
                residuals - backcast
            ) * input_mask  # Adjust residuals (shape: (batch_size, time_steps))
            forecast = (
                forecast + block_forecast
            )  # Accumulate forecast (shape: (batch_size, forecast_size))

        return forecast  # Final forecast (shape: (batch_size, forecast_size))


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        """
        Forward pass for the generic basis function.

        :param theta: Tensor containing parameters for backcast and forecast (shape: (batch_size, theta_size))
        :return: backcast and forecast tensors (shapes: (batch_size, backcast_size), (batch_size, forecast_size))
        """
        # Split theta into backcast and forecast parameters
        backcast = theta[:, : self.backcast_size]  # (batch_size, backcast_size)
        forecast = theta[:, -self.forecast_size :]  # (batch_size, forecast_size)
        return backcast, forecast


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        self.polynomial_size = (
            degree_of_polynomial + 1
        )  # Degree of polynomial with constant term

        # Time grid for backcast and forecast (size: backcast_size, polynomial_size)
        self.backcast_time = t.nn.Parameter(
            t.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=np.float64) / backcast_size,
                            i,
                        )[None, :]
                        for i in range(self.polynomial_size)
                    ]
                ),
                dtype=t.float32,
            ),
            requires_grad=False,
        )

        self.forecast_time = t.nn.Parameter(
            t.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=np.float64) / forecast_size,
                            i,
                        )[None, :]
                        for i in range(self.polynomial_size)
                    ]
                ),
                dtype=t.float32,
            ),
            requires_grad=False,
        )

    def forward(self, theta: t.Tensor):
        """
        Forward pass for the Trend Basis function.

        :param theta: Tensor containing parameters for trend (shape: (batch_size, theta_size))
        :return: backcast and forecast tensors (shapes: (batch_size, backcast_size), (batch_size, forecast_size))
        """
        backcast = t.einsum(
            "bp,pt->bt", theta[:, self.polynomial_size :], self.backcast_time
        )  # (batch_size, backcast_size)
        forecast = t.einsum(
            "bp,pt->bt", theta[:, : self.polynomial_size], self.forecast_time
        )  # (batch_size, forecast_size)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()

        self.frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32)
            / harmonics,
        )[None, :]

        # Time grids for backcast and forecast (sizes: backcast_size, forecast_size, harmonics)
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size)
            * self.frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size)
            * self.frequency
        )

        self.backcast_cos_template = t.nn.Parameter(
            t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
            requires_grad=False,
        )
        self.backcast_sin_template = t.nn.Parameter(
            t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
            requires_grad=False,
        )
        self.forecast_cos_template = t.nn.Parameter(
            t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
            requires_grad=False,
        )
        self.forecast_sin_template = t.nn.Parameter(
            t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
            requires_grad=False,
        )

    def forward(self, theta: t.Tensor):
        """
        Forward pass for the Seasonality Basis function.

        :param theta: Tensor containing parameters for seasonality (shape: (batch_size, theta_size))
        :return: backcast and forecast tensors (shapes: (batch_size, backcast_size), (batch_size, forecast_size))
        """
        params_per_harmonic = theta.shape[1] // 4  # Divide parameters into harmonics

        backcast_harmonics_cos = t.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic],
            self.backcast_cos_template,
        )  # (batch_size, backcast_size)
        backcast_harmonics_sin = t.einsum(
            "bp,pt->bt", theta[:, 3 * params_per_harmonic :], self.backcast_sin_template
        )  # (batch_size, backcast_size)
        backcast = (
            backcast_harmonics_sin + backcast_harmonics_cos
        )  # (batch_size, backcast_size)

        forecast_harmonics_cos = t.einsum(
            "bp,pt->bt", theta[:, :params_per_harmonic], self.forecast_cos_template
        )  # (batch_size, forecast_size)
        forecast_harmonics_sin = t.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic : 2 * params_per_harmonic],
            self.forecast_sin_template,
        )  # (batch_size, forecast_size)
        forecast = (
            forecast_harmonics_sin + forecast_harmonics_cos
        )  # (batch_size, forecast_size)

        return backcast, forecast


class NBEATS_I(t.nn.Module):
    def __init__(self, configs):
        """
        N-BEATS interpretable model using trend and seasonality basis with parameters from a config.

        :param input_size: Length of the input time series for each time step.
        :param pred_len: Length of the forecast horizon (number of future time steps to predict).
        :param trend_blocks: Number of trend blocks (each block learns a separate trend component).
        :param trend_layers: Number of fully connected layers in each trend block.
        :param trend_layer_size: Number of neurons in each hidden layer of the trend block.
        :param degree_of_polynomial: The degree of the polynomial used in the trend basis function.
        :param seasonality_blocks: Number of seasonality blocks (each block models a different seasonality component).
        :param seasonality_layers: Number of fully connected layers in each seasonality block.
        :param seasonality_layer_size: Number of neurons in each hidden layer of the seasonality block.
        :param num_of_harmonics: The number of harmonics used in the seasonality basis function.
        """
        super().__init__()

        # Extracting parameters from the config
        self.input_size = configs.seq_len * configs.in_features
        self.output_size = configs.pred_len
        self.pred_len = configs.pred_len * configs.in_features
        self.trend_blocks = configs.trend_blocks
        self.trend_layers = configs.trend_layers
        self.trend_layer_size = configs.trend_layer_size
        self.degree_of_polynomial = configs.degree_of_polynomial
        self.seasonality_blocks = configs.seasonality_blocks
        self.seasonality_layers = configs.seasonality_layers
        self.seasonality_layer_size = configs.seasonality_layer_size
        self.num_of_harmonics = configs.num_of_harmonics

        # Trend and Seasonality blocks setup using the provided functions
        self.trend_block = NBeatsBlock(
            input_size=self.input_size,
            theta_size=2 * (self.degree_of_polynomial + 1),
            basis_function=TrendBasis(
                degree_of_polynomial=self.degree_of_polynomial,
                backcast_size=self.input_size,
                forecast_size=self.pred_len,
            ),
            layers=self.trend_layers,
            layer_size=self.trend_layer_size,
        )

        self.seasonality_block = NBeatsBlock(
            input_size=self.input_size,
            theta_size=4
            * int(
                np.ceil(self.num_of_harmonics / 2 * self.pred_len)
                - (self.num_of_harmonics - 1)
            ),
            basis_function=SeasonalityBasis(
                harmonics=self.num_of_harmonics,
                backcast_size=self.input_size,
                forecast_size=self.pred_len,
            ),
            layers=self.seasonality_layers,
            layer_size=self.seasonality_layer_size,
        )

        # Create the nbeatsi module list using the config values
        self.nbeats_i = NBeats(
            t.nn.ModuleList(
                [self.trend_block for _ in range(self.trend_blocks)]
                + [self.seasonality_block for _ in range(self.seasonality_blocks)]
            )
        )

    def forward(self, x: t.Tensor):
        """
        Forward pass for N-BEATS interpretable model.

        :param x: Input tensor of shape (batch_size, seq_len, in_features)
        :return: Forecast tensor of shape (batch_size, pred_len, in_features)
        """
        batch_size, seq_len, in_features = x.shape

        # (batch_size, seq_len * in_features)
        x = x.view(
            batch_size, -1
        )  # Flatten the input tensor to combine seq_len and in_features

        input_mask = t.ones_like(x)  # (batch_size, seq_len * in_features)
        forecast = self.nbeats_i(x, input_mask)

        forecast = forecast.view(
            batch_size, self.output_size, in_features
        )  # (batch_size, pred_len, in_features)

        return forecast  # (batch_size, pred_len, in_features)


class NBEATS_G(t.nn.Module):
    def __init__(self, configs):
        super(NBEATS_G, self).__init__()
        """
        :param input_size: Length of the input time series for each time step.
        :param pred_len: Length of the forecast horizon (number of future time steps to predict).
        :param stacks: Number of stacks (groups of blocks) in the N-BEATS architecture.
        :param layers: Number of fully connected layers in each block.
        :param layer_size: Number of neurons in each hidden layer.
        """
        self.input_size = configs["seq_len"] * configs["enc_in"]
        self.output_size = configs["pred_len"]
        self.pred_len = configs["pred_len"] * configs["enc_in"]
        self.stacks = configs["Nbeats_stacks"]
        self.layers = configs["num_layers"]
        self.layer_size = configs["hidden_dim"]

        self.nbeats_g = NBeats(
            t.nn.ModuleList(
                [
                    NBeatsBlock(
                        input_size=self.input_size,
                        theta_size=self.input_size + self.pred_len,
                        basis_function=GenericBasis(
                            backcast_size=self.input_size, forecast_size=self.pred_len
                        ),
                        layers=self.layers,
                        layer_size=self.layer_size,
                    )
                    for _ in range(self.stacks)
                ]
            )
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Forward pass for N-BEATS Generic model with combined input (seq_len * in_features).

        :param x: Input tensor of shape (batch_size, seq_len, in_features)
        :return: Forecast tensor of shape (batch_size, pred_len, in_features)
        """
        batch_size, seq_len, in_features = x.shape

        # (batch_size, seq_len * in_features)
        x = x.view(
            batch_size, -1
        )  # Flatten the input tensor to combine seq_len and in_features

        input_mask = t.ones_like(x)  # (batch_size, seq_len * in_features)
        forecast = self.nbeats_g(x, input_mask)  # (batch_size, pred_len * in_features)

        forecast = forecast.view(
            batch_size, self.output_size, in_features
        )  # (batch_size, pred_len, in_features)

        return forecast  # (batch_size, pred_len, in_features)


class Nbeats(t.nn.Module):
    def __init__(self, configs):
        """
        Main N-BEATS model that selects either a generative or interpretable model based on configuration.

        :param configs: Dictionary or object containing model configuration.
        """
        super().__init__()
        self.fcst_type = configs["fcst_type"]
        # Store the model type from the configuration
        self.model_type = configs["Nbeats_model_type"]
        # Choose the model type based on the configuration
        if self.model_type == "nbeats_g":
            # Instantiate the generative model
            self.model = NBEATS_G(configs)
        elif self.model_type == "nbeats_i":
            # Instantiate the interpretable model
            self.model = NBEATS_I(configs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input data to the model.
        :return: Model output.
        """
        x = self.model(x)
        if self.fcst_type == "MS":
            return x[:, :, 0].unsqueeze(-1)
        else:
            return x
