import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import os
import sys
import time
import threading
from argparse import ArgumentParser
from collections import deque, namedtuple
import random
import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[Warning] psutil not available - parent monitoring disabled")

# Make sure these are imported from your myTeam.py file
from myTeam import (
    SmurphGNN,
    SmurphAgentConfig,
    Transition,
)

# Batch container for training (different from experience Transition)
BatchTransition = namedtuple(
    "BatchTransition", ("state", "action", "reward", "next_state", "done")
)

DQN_CONFIG = {
    "BUFFER_SIZE": 5000,  # Max number of experiences to store
    "BATCH_SIZE": 64,  # Number of experiences to sample for each training step
    "EPSILON_START": 0.9,  # Starting exploration rate
    "COPY_FREQUENCY": 500,
    "SAVE_FREQUENCY": 50,
}


def exit_if_parent_dead():
    """Monitor parent process and exit if it dies (prevents orphaned trainers)."""
    if not PSUTIL_AVAILABLE:
        return

    try:
        parent = psutil.Process(os.getppid())
        parent_pid = parent.pid
        print(f"[ParentMonitor] Monitoring parent PID {parent_pid}")

        while True:
            if not parent.is_running():
                print("[ParentMonitor] Parent process died, exiting trainer.")
                os._exit(0)
            time.sleep(5)
    except Exception as e:
        print(f"[ParentMonitor] Error monitoring parent: {e}")
        # Don't exit - let the trainer continue if monitoring fails


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        transition = Transition(state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # importance-sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        # Convert list of Transitions to BatchTransition
        batch = BatchTransition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


def get_weights_path(agentId: int):
    return f"weights/{agentId}.pt"


def get_config_path(agentId: int):
    return f"configs/{agentId}.pt"


def main(agentId: int):
    # Start parent monitoring thread to prevent orphaned processes
    monitor_thread = threading.Thread(target=exit_if_parent_dead, daemon=True)
    monitor_thread.start()

    config = torch.load(get_config_path(agentId))
    assert type(config).__name__ == "SmurphAgentConfig"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    learner_model = SmurphGNN(num_node_features=11, num_actions=5).to(device)
    learner_model.load_state_dict(torch.load(get_weights_path(agentId)))
    target_net = SmurphGNN(num_node_features=11, num_actions=5).to(device)
    target_net.load_state_dict(learner_model.state_dict())
    optimizer = optim.Adam(learner_model.parameters(), lr=config.learning_rate)
    replay_buffer = PrioritizedReplayBuffer(DQN_CONFIG["BUFFER_SIZE"])
    step_count = 0

    print(f"[Trainer {agentId}] Starting training loop...")
    last_exp_count = 0
    last_log_time = time.time()

    while True:  # Loop forever, or for a fixed number of generations
        # --- 1. Learner: Collect Experiences ---
        exp_files = [
            f for f in os.listdir(f"experiences/{agentId}") if f.endswith(".pt")
        ]
        exp_count_this_round = 0

        for f in exp_files:
            filepath = os.path.join(f"experiences/{agentId}", f)
            try:
                new_exps = torch.load(filepath)
                assert isinstance(new_exps, list)
                for exp in new_exps:
                    # Experiences are saved as tuples (state, action, reward, next_state, done)
                    if isinstance(exp, tuple) and len(exp) == 5:
                        state, action, reward, next_state, done = exp
                        replay_buffer.push(state, action, reward, next_state, done)
                        exp_count_this_round += 1
                    else:
                        print(
                            f"[Trainer {agentId}] Unknown experience format: {type(exp)}"
                        )
                os.remove(filepath)
            except Exception as e:
                print(f"[Trainer {agentId}] Could not load or delete {filepath}: {e}")

        # Log experience collection every 30 seconds
        if exp_count_this_round > 0:
            last_exp_count += exp_count_this_round

        if time.time() - last_log_time > 30:
            print(
                f"[Trainer {agentId}] Buffer: {len(replay_buffer)}/{DQN_CONFIG['BUFFER_SIZE']} | "
                f"New experiences: {last_exp_count} | Steps: {step_count}"
            )
            last_exp_count = 0
            last_log_time = time.time()

        # --- 2. Learner: Train the Model ---
        if len(replay_buffer) > DQN_CONFIG["BATCH_SIZE"]:
            learner_model.train()
            batch, indices, weights = replay_buffer.sample(DQN_CONFIG["BATCH_SIZE"])

            state_batch = Batch.from_data_list(batch.state).to(device)
            action_batch = torch.tensor(batch.action, dtype=torch.long, device=device)
            reward_batch = torch.tensor(
                batch.reward, dtype=torch.float32, device=device
            )

            non_final_mask = torch.tensor(
                [s is not None for s in batch.next_state],
                dtype=torch.bool,
                device=device,
            )
            non_final_next_states = [s for s in batch.next_state if s is not None]

            # Compute current Q(s,a)
            q_pred = (
                learner_model(state_batch)
                .gather(1, action_batch.unsqueeze(1))
                .squeeze()
            )

            # Compute target Double DQN values
            next_q_values = torch.zeros(DQN_CONFIG["BATCH_SIZE"], device=device)
            if len(non_final_next_states) > 0:
                next_batch = Batch.from_data_list(non_final_next_states).to(device)
                with torch.no_grad():
                    next_actions = learner_model(next_batch).argmax(1)
                    next_q_target = (
                        target_net(next_batch)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze()
                    )
                next_q_values[non_final_mask] = next_q_target

            q_target = reward_batch + config.gamma * next_q_values

            # TD error and prioritized update
            td_errors = torch.abs(q_target - q_pred).detach().cpu().numpy()
            replay_buffer.update_priorities(indices, td_errors + 1e-5)

            # Weighted loss
            weights = weights.to(device)
            loss = (
                weights * F.smooth_l1_loss(q_pred, q_target, reduction="none")
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner_model.parameters(), 10.0)
            optimizer.step()

            if step_count % DQN_CONFIG["COPY_FREQUENCY"] == 0:
                target_net.load_state_dict(learner_model.state_dict())

            if step_count % DQN_CONFIG["SAVE_FREQUENCY"] == 0:
                torch.save(learner_model.state_dict(), get_weights_path(agentId))
                print(
                    f"[Trainer {agentId}] Saved weights at step {step_count} | Loss: {loss.item():.4f}"
                )

            step_count += 1

        time.sleep(0.1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--agentId", type=int)
    args = parser.parse_args()

    main(args.agentId)
