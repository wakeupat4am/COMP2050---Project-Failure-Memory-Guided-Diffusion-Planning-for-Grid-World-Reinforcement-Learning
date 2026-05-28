from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.embedding_dim:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=emb.device)], dim=1)
        return emb


class ConditionalDenoisingMLP(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim + condition_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time_features = self.time_embedding(t)
        features = torch.cat([x_t, condition, time_features], dim=1)
        return self.network(features)


class DiffusionActionModel:
    def __init__(
        self,
        horizon: int,
        action_dim: int = 4,
        condition_dim: int = 4,
        hidden_dim: int = 128,
        diffusion_steps: int = 50,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.input_dim = horizon * action_dim
        self.condition_dim = condition_dim
        self.diffusion_steps = diffusion_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConditionalDenoisingMLP(
            input_dim=self.input_dim,
            condition_dim=self.condition_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def training_step(self, x0: torch.Tensor, condition: torch.Tensor) -> float:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted_noise = self.model(x_t, t, condition)
        loss = self.loss_fn(predicted_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fit(self, action_vectors: np.ndarray, conditions: np.ndarray, epochs: int = 30, batch_size: int = 64):
        dataset = TensorDataset(
            torch.tensor(action_vectors, dtype=torch.float32),
            torch.tensor(conditions, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        history = []
        self.model.train()
        for _ in range(epochs):
            batch_losses = []
            for x0, condition in loader:
                x0 = x0.to(self.device)
                condition = condition.to(self.device)
                batch_losses.append(self.training_step(x0, condition))
            history.append(float(np.mean(batch_losses)))
        return history

    @torch.no_grad()
    def sample(self, condition, num_samples: int, horizon: int | None = None):
        self.model.eval()
        condition = torch.tensor(condition, dtype=torch.float32, device=self.device)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        condition = condition.repeat(num_samples, 1)

        x_t = torch.randn((num_samples, self.input_dim), device=self.device)
        for t_idx in reversed(range(self.diffusion_steps)):
            t = torch.full((num_samples,), t_idx, dtype=torch.long, device=self.device)
            predicted_noise = self.model(x_t, t, condition)
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bars[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            )
            if t_idx > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        actions = x_t.view(num_samples, self.horizon, self.action_dim).argmax(dim=-1)
        return actions.cpu().numpy()
