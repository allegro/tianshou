from typing import Any, Dict

import torch

from tianshou.data import Batch
from tianshou.policy import SACPolicy


class CQLPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = 10
        self.lagrange_budget = False

    @staticmethod
    def _expand_obs(obs: torch.Tensor, multiplier: int):
        obs = obs.unsqueeze(1).expand(-1, multiplier, -1).reshape(-1, obs.shape[1])
        return obs

    def _sample_q(self, batch, critic):
        # TODO: log difference in Q between random actions and actions from replay buffer
        expanded_obs = self._expand_obs(batch.obs, self.num_samples)
        expanded_obs_next = self._expand_obs(batch.obs_next, self.num_samples)

        expanded_batch = Batch(obs=expanded_obs, obs_next=expanded_obs_next)
        with torch.no_grad():
            expanded_obs_actions = self(expanded_batch, input="obs")
            expanded_obs_next_actions = self(expanded_batch, input="obs_next")
        act, act_log_prob = expanded_obs_actions.act, expanded_obs_actions.log_prob
        act_next, act_next_log_prob = (
            expanded_obs_next_actions.act,
            expanded_obs_next_actions.log_prob,
        )

        random_actions = (
            torch.FloatTensor(act.shape).uniform_(-1, 1).to(device=self.device)
        )
        obs_q = critic(expanded_obs, act)
        next_obs_q = critic(expanded_obs, act_next)
        random_q = critic(expanded_obs, random_actions)

        # Importance sampling
        random_density = torch.ones(
            len(random_actions), device=self.device
        ) * torch.log(0.5 ** random_actions.shape[-1])
        obs_q = obs_q - act_log_prob
        next_obs_q = next_obs_q - act_next_log_prob
        random_q = random_q - random_density

        obs_q = obs_q.reshape(-1, self.num_samples)
        next_obs_q = next_obs_q.reshape(-1, self.num_samples)
        random_q = random_q.reshape(-1, self.num_samples)
        sampled_q = torch.cat([obs_q, next_obs_q, random_q], axis=1)
        return sampled_q

    def _cql_optimizer(
        self, batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        mse_loss = (td.pow(2) * weight).mean()

        sampled_q = self._sample_q(batch, critic)
        q_all = torch.cat([sampled_q, current_q.unsqueeze(1)], dim=1)
        loss_cql = torch.logsumexp(q_all, dim=1).mean() - current_q.mean()
        if self.lagrange_budget:
            raise NotImplementedError
        else:
            loss_cql = self.cql_weight * loss_cql

        critic_loss = mse_loss + loss_cql
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss = self._cql_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss = self._cql_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a).flatten()
        current_q2a = self.critic2(batch.obs, a).flatten()
        actor_loss = (
            self._alpha * obs_result.log_prob.flatten()
            - torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
