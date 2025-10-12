import torch
from typing import List
import random, math, time, uuid, json, os, subprocess, shutil, sys, atexit, signal
from myTeam import SmurphAgentConfig, RewardWeights, SmurphGNN
from dataclasses import replace

# Global registry of trainer subprocesses for cleanup
trainer_procs = {}

PBT_CFG = {
    "population_size": 12,
    "concurrent_games": 4,
    "games_per_match": 3,
    "exploit_interval": 200,
    "burn_in_games": 20,
    "elo_K": 32,
    "sigma_winprob": 1 / 6,
    "seed": 42,
}


def logu(a, b):  # log-uniform
    return 10 ** random.uniform(math.log10(a), math.log10(b))


def signed_logu(sign, a=0.1, b=10.0):
    return (1 if sign > 0 else -1) * logu(a, b)


# Example mapping (choose signs by event polarity)
DEFAULT_REWARD_WEIGHTS = RewardWeights(
    score_change=signed_logu(+1),
    scored_points=signed_logu(+1),
    lost_points=signed_logu(-1),
    eaten_as_scared_ghost=signed_logu(-1),
    ate_food=signed_logu(+1),
    ate_capsule=signed_logu(+1),
    teammate_scored_points=signed_logu(+1),
    teammate_ate_food=signed_logu(+1),
    saved_points=signed_logu(+1),
    ate_scared_ghost=signed_logu(+1),
    food_eaten_by_opponent=signed_logu(-1),
    capsule_eaten_by_opponent=signed_logu(-1),
    time_penalty=signed_logu(-1),
)
DEFAULT_CONFIG = SmurphAgentConfig(
    reward_weights=DEFAULT_REWARD_WEIGHTS,
    learning_rate=logu(1e-5, 1e-3),
    gamma=0.99,
    epsilon_start=logu(0.1, 0.8),
    epsilon_decay_rate=random.uniform(2000, 10000),
    epsilon_min=0.01,
    games_played=0,
)


def elo_expected_1v1(eloA, eloB):
    return 1.0 / (1.0 + 10 ** ((eloB - eloA) / 400.0))


def update_elo_1v1(ratings, A, B, scoreA, K=32):
    expA = elo_expected_1v1(ratings[A], ratings[B])
    deltaA = K * (scoreA - expA)
    ratings[A] += deltaA
    ratings[B] -= deltaA


def _gauss_centered_at_0p5(p, sigma):
    z = (p - 0.5) / sigma
    return math.exp(-0.5 * z * z)


def sample_pair(population, ratings, sigma=1 / 6, tries=20):
    if len(population) < 2:
        raise ValueError("Need at least 2 agents.")
    A = random.choice(population)
    candidates = [x for x in population if x != A]
    best = None
    best_w = -1
    for _ in range(min(tries, len(candidates))):
        B = random.choice(candidates)
        pA = elo_expected_1v1(ratings[A], ratings[B])
        w = _gauss_centered_at_0p5(pA, sigma)
        if w > best_w:
            best_w, best = w, (A, B)
    return best if best else (A, random.choice(candidates))


def cleanup_trainers():
    """Terminate all trainer subprocesses on exit."""
    if not trainer_procs:
        return
    print("\n[Cleanup] Terminating all trainer subprocesses...")
    for aid, proc in trainer_procs.items():
        if proc.poll() is None:  # still running
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"[Cleanup] Terminated trainer {aid}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"[Cleanup] Force killed trainer {aid}")
            except Exception as e:
                print(f"[Cleanup] Error terminating trainer {aid}: {e}")
    print("[Cleanup] All trainer processes terminated.")


def handle_sigterm(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    print(f"\n[Signal] Received signal {signum}, shutting down...")
    cleanup_trainers()
    sys.exit(0)


def launch_trainers(agent_ids):
    procs = {}
    for i in agent_ids:
        # Launch your learner; ensure it runs forever and watches experiences/{i}
        p = subprocess.Popen(
            [sys.executable, "policy_trainer.py", "--agentId", str(i)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        procs[i] = p
        print(f"[launch_trainers] Started trainer for agent {i} (PID: {p.pid})")
    # Register for global cleanup
    trainer_procs.update(procs)
    return procs


def run_game_homogeneous(A_id, B_id, num_games=1, layout=None, timeout=120):
    """
    Runs capture.py with:
      -r myTeam -b myTeam
      --redOpts agentId=A_id
      --blueOpts agentId=B_id
      -n num_games
      --results-out <tempfile.json>
    Returns parsed dict or None on failure.
    """
    os.makedirs("runners", exist_ok=True)
    results_path = os.path.join("runners", f"results_{uuid.uuid4().hex}.json")

    cmd = [
        sys.executable,  # Use the same Python interpreter as the parent process
        "capture.py",
        "-r",
        "myTeam",
        "-b",
        "myTeam",
        "--redOpts",
        f"agentId={A_id},mode=training",
        "--blueOpts",
        f"agentId={B_id},mode=training",
        "-q",
        "-n",
        str(num_games),
        "-x",
        "0",
        "--results-out",
        results_path,
    ]
    if layout:
        cmd += ["-l", layout]

    try:
        print(f"  [Game] Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)

        if result.returncode != 0:
            print(f"  [Game] Non-zero exit code: {result.returncode}")
            print(f"  [Game] STDOUT: {result.stdout[:500]}")
            print(f"  [Game] STDERR: {result.stderr[:500]}")
            data = None
        else:
            print(f"  [Game] Completed successfully, reading results...")
            with open(results_path, "r") as f:
                data = json.load(f)
    except subprocess.TimeoutExpired:
        print(f"  [Game] TIMEOUT after {timeout}s")
        data = None
    except subprocess.CalledProcessError as e:
        print(f"  [Game] subprocess failed:", e.output.decode("utf-8", errors="ignore")[:500])
        data = None
    except Exception as e:
        print(f"  [Game] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        data = None
    finally:
        # keep the JSON if you want historical logs:
        try:
            os.remove(results_path)
        except:
            pass

    return data


def apply_elo_from_results(ratings, red_id, blue_id, results_dict, K=32):
    if not results_dict:
        return 0
    outcomes = results_dict.get("results", [])
    updated = 0
    for r in outcomes:
        if r == "Red":
            scoreA = 1.0
        elif r == "Blue":
            scoreA = 0.0
        else:
            scoreA = 0.5  # "Tie"
        update_elo_1v1(ratings, red_id, blue_id, scoreA, K=K)
        updated += 1
    return updated  # number of games applied


def init_population(default_cfg: SmurphAgentConfig, reset=False) -> List[int]:
    if reset:
        print("[init_population] Resetting population from scratch...")
        shutil.rmtree("weights", ignore_errors=True)
        shutil.rmtree("configs", ignore_errors=True)
        shutil.rmtree("experiences", ignore_errors=True)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("experiences", exist_ok=True)

    pop = []
    for i in range(PBT_CFG["population_size"]):
        cfg_path = f"configs/{i}.pt"
        weights_path = f"weights/{i}.pt"
        exp_dir = f"experiences/{i}"

        # --- Case 1: Existing agent → resume
        if os.path.exists(cfg_path) and os.path.exists(weights_path):
            print(f"[init_population] Resuming agent {i}")
            pop.append(i)
            # Make sure experience dir still exists
            os.makedirs(exp_dir, exist_ok=True)
            continue

        # --- Case 2: Create new agent
        print(f"[init_population] Creating new agent {i}")
        cfg = replace(default_cfg)

        # mild randomisation of rewards/hparams at init (log-uniform-ish)
        def jitter(x, lo=0.5, hi=2.0):
            return float(x * (10 ** random.uniform(math.log10(lo), math.log10(hi))))

        rw = cfg.reward_weights
        rw = RewardWeights(
            score_change=rw.score_change,
            scored_points=jitter(rw.scored_points),
            lost_points=jitter(rw.lost_points),
            eaten_as_scared_ghost=jitter(rw.eaten_as_scared_ghost),
            ate_food=jitter(rw.ate_food),
            ate_capsule=jitter(rw.ate_capsule),
            teammate_scored_points=jitter(rw.teammate_scored_points),
            teammate_ate_food=jitter(rw.teammate_ate_food),
            saved_points=jitter(rw.saved_points),
            ate_scared_ghost=jitter(rw.ate_scared_ghost),
            food_eaten_by_opponent=jitter(rw.food_eaten_by_opponent),
            capsule_eaten_by_opponent=jitter(rw.capsule_eaten_by_opponent),
            time_penalty=rw.time_penalty,  # often fixed
        )
        cfg = replace(cfg, reward_weights=rw)
        torch.save(cfg, cfg_path)

        # Initialize model (fresh or from default init)
        model = SmurphGNN(num_node_features=11, num_actions=5, hidden_channels=32)
        torch.save(model.state_dict(), weights_path)

        os.makedirs(exp_dir, exist_ok=True)
        pop.append(i)

    return pop


def mutate_config(cfg: SmurphAgentConfig):
    def maybe_scale(x, p=0.05):
        if random.random() < p:  # +20%
            return x * 1.2
        if random.random() < p:  # -20%
            return x * 0.8
        return x

    rw = cfg.reward_weights
    rw = RewardWeights(
        score_change=rw.score_change,
        scored_points=maybe_scale(rw.scored_points),
        lost_points=maybe_scale(rw.lost_points),
        eaten_as_scared_ghost=maybe_scale(rw.eaten_as_scared_ghost),
        ate_food=maybe_scale(rw.ate_food),
        ate_capsule=maybe_scale(rw.ate_capsule),
        saved_points=maybe_scale(rw.saved_points),
        ate_scared_ghost=maybe_scale(rw.ate_scared_ghost),
        teammate_scored_points=maybe_scale(rw.teammate_scored_points),
        teammate_ate_food=maybe_scale(rw.teammate_ate_food),
        food_eaten_by_opponent=maybe_scale(rw.food_eaten_by_opponent),
        capsule_eaten_by_opponent=maybe_scale(rw.capsule_eaten_by_opponent),
        time_penalty=rw.time_penalty,
    )
    cfg = replace(
        cfg,
        reward_weights=rw,
        learning_rate=maybe_scale(cfg.learning_rate),
        gamma=cfg.gamma,  # often fixed
        epsilon_decay_rate=maybe_scale(cfg.epsilon_decay_rate),
    )
    return cfg


def exploit_and_explore(ratings, burn_until):
    N = len(ratings)
    sorted_ids = sorted(ratings.keys(), key=lambda k: ratings[k], reverse=True)
    top_k = set(sorted_ids[: max(1, int(PBT_CFG["top_frac"] * N))])
    bot_k = sorted_ids[-max(1, int(PBT_CFG["bottom_frac"] * N)) :]
    for loser in bot_k:
        cfg = torch.load(f"configs/{loser}.pt")
        assert type(cfg).__name__ == "SmurphAgentConfig"
        if cfg.games_played < PBT_CFG["burn_in_games"]:
            continue  # skip fresh agents
        donor = random.choice(list(top_k))
        if donor == loser:
            continue
        # copy weights + config
        shutil.copyfile(f"weights/{donor}.pt", f"weights/{loser}.pt")
        cfg = torch.load(f"configs/{donor}.pt")
        cfg = mutate_config(cfg)
        cfg = replace(cfg, games_played=0)
        torch.save(cfg, f"configs/{loser}.pt")
        burn_until[loser] = burn_until.get(loser, 0) + PBT_CFG["burn_in_games"]


def population_based_training(default_cfg: SmurphAgentConfig, reset=False):
    population = init_population(default_cfg, reset=reset)

    # Load or initialize ELO ratings
    os.makedirs("logs", exist_ok=True)
    elo_path = "logs/elo.json"
    if os.path.exists(elo_path) and not reset:
        with open(elo_path, "r") as f:
            ratings = json.load(f)
            ratings = {int(k): float(v) for k, v in ratings.items()}
        print(f"[PBT] Loaded ELO ratings for {len(ratings)} agents.")
    else:
        ratings = {i: 1200.0 for i in population}
        print(f"[PBT] Initialized fresh ELO ratings.")

    burn_until = {i: 0 for i in population}

    print(f"\n=== PBT Started ===")
    print(f"Population: {len(population)} agents")
    print(f"Concurrent games: {PBT_CFG['concurrent_games']}")
    print(f"Games per match: {PBT_CFG['games_per_match']}")
    print(f"Exploit interval: {PBT_CFG['exploit_interval']} games\n")

    _ = launch_trainers(population)

    total_games = 0
    generation = 0
    while True:
        # schedule pairs
        pairs = []
        for _ in range(PBT_CFG["concurrent_games"]):
            A, B = sample_pair(population, ratings, sigma=PBT_CFG["sigma_winprob"])
            # randomise sides to avoid bias
            if random.random() < 0.5:
                pairs.append((A, B))  # red=A, blue=B
            else:
                pairs.append((B, A))  # red=B, blue=A

        # Launch games in parallel
        import concurrent.futures

        print(f"  [Game] Launching {len(pairs)} games in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as executor:
            # Submit all games
            futures = {}
            for red_id, blue_id in pairs:
                print(f"  [Game] Queueing: Agent {red_id} (Red) vs Agent {blue_id} (Blue)")
                future = executor.submit(
                    run_game_homogeneous,
                    red_id, blue_id,
                    num_games=PBT_CFG["games_per_match"]
                )
                futures[future] = (red_id, blue_id)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                red_id, blue_id = futures[future]
                try:
                    res = future.result()
                    if res:
                        print(f"  [Game] Completed: Agent {red_id} vs {blue_id} | Results: {res.get('results', [])} | Scores: {res.get('scores', [])}")
                    else:
                        print(f"  [Game] FAILED: Agent {red_id} vs {blue_id} - no results returned")

                    ng = apply_elo_from_results(
                        ratings, red_id, blue_id, res, K=PBT_CFG["elo_K"]
                    )
                    total_games += ng

                    increment_games_played(red_id, ng)
                    increment_games_played(blue_id, ng)

                    # burn-in bookkeeping
                    if burn_until[red_id] > 0:
                        burn_until[red_id] = max(0, burn_until[red_id] - ng)
                    if burn_until[blue_id] > 0:
                        burn_until[blue_id] = max(0, burn_until[blue_id] - ng)
                except Exception as e:
                    print(f"  [Game] Exception in game {red_id} vs {blue_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Print progress after each round of concurrent games
        print(f"[Gen {generation:4d}] Games: {total_games:5d} | ", end="")
        sorted_agents = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_agents[:3]
        print(f"Top3: {top3[0][0]}({top3[0][1]:.0f}) {top3[1][0]}({top3[1][1]:.0f}) {top3[2][0]}({top3[2][1]:.0f})")

        generation += 1

        # PBT exploit/explore periodically
        if total_games and total_games % PBT_CFG["exploit_interval"] == 0:
            print(f"\n*** EXPLOIT/EXPLORE at {total_games} games ***")
            exploit_and_explore(ratings, burn_until)
            print("Leaderboard after E&E:")
            sorted_agents = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
            for rank, (agent_id, elo) in enumerate(sorted_agents[:5], 1):
                cfg = torch.load(f"configs/{agent_id}.pt")
                print(f"  {rank}. Agent {agent_id:2d}: ELO={elo:7.1f} Games={cfg.games_played:4d}")
            print()

        if total_games % 50 == 0 or total_games % PBT_CFG["games_per_match"] == 0:
            os.makedirs("logs", exist_ok=True)
            # Save ELO ratings for resume
            with open("logs/elo.json", "w") as f:
                json.dump(ratings, f, indent=2)
            # Save detailed status
            with open("logs/pbt_status.json", "w") as f:
                json.dump({"games": total_games, "ratings": ratings, "generation": generation}, f, indent=2)

        time.sleep(0.2)


def increment_games_played(agent_id: int, n: int = 1):
    cfg_path = f"configs/{agent_id}.pt"
    cfg = torch.load(cfg_path)
    cfg.games_played += n
    torch.save(cfg, cfg_path)


if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_trainers)
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    # Pass --reset flag to start from scratch
    reset = "--reset" in sys.argv
    population_based_training(DEFAULT_CONFIG, reset=reset)
