import torch
from rsl_rl.utils import split_and_pad_trajectories


class CMOStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.costs = None
            self.dones = None
            self.values = None
            self.cost_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
            self,
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            device="cpu"
    ):
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        # Core data
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
            )
        else:
            self.privileged_observations = None
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.uint8, device=device)

        # Value function
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        # Actor stats
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Rollout buffer overflow!")
        
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.costs[self.step].copy_(transition.costs.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.values[self.step].copy_(transition.values)
        self.cost_values[self.step].copy_(transition.cost_values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + gamma * next_values * next_is_not_terminal - self.values[step]
            advantage = delta + gamma * lam * next_is_not_terminal * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_cost_returns(self, last_cost_values, gamma, lam, normalize_advantage=True):
        gae = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_cost = last_cost_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.costs[step] + gamma * next_cost * next_is_not_terminal - self.cost_values[step]
            gae = delta + gamma * lam * next_is_not_terminal * gae
            self.cost_returns[step] = gae + self.cost_values[step]
        self.cost_advantages = self.cost_returns - self.cost_values
        if normalize_advantage:
            self.cost_advantages = (self.cost_advantages - self.cost_advantages.mean()) / (self.cost_advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size, device=self.device)

        # Flatten buffers
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        costs = self.costs.flatten(0, 1)

        values = self.values.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]
                costs_batch = costs[batch_idx]
                target_values_batch = values[batch_idx]
                cost_values_batch = cost_values[batch_idx]
                returns_batch = returns[batch_idx]
                cost_returns_batch = cost_returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                cost_advantages_batch = cost_advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]         

                yield obs_batch, privileged_observations_batch, actions_batch, costs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    cost_values_batch,
                    cost_advantages_batch,
                ), cost_returns_batch, None

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_observations_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                costs_batch = self.costs[:, start:stop]
                target_values_batch = self.values[:, start:stop]
                cost_values_batch = self.cost_values[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                cost_returns_batch = self.cost_returns[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                cost_advantages_batch = self.cost_advantages[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]

                yield obs_batch, privileged_observations_batch, actions_batch, costs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    cost_values_batch,
                    cost_advantages_batch,
                ), cost_returns_batch, masks_batch

                first_traj = last_traj
