"""Métricas comunes: retorno, máximo drawdown y costos de transacción."""

import numpy as np


def summarize(env, info):
    initial = env.initial_balance
    return {
        "final_net_worth": info["net_worth"],
        "return_pct": ((info["net_worth"] - initial) / initial) * 100,
        "max_drawdown_pct": info["max_drawdown"] * 100,
        "costs": info["total_trade_cost"],
    }


def run_episode(env, action_fn, seed=42):
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    equity = [env.net_worth]
    positions = [env.current_position]
    done = truncated = False
    while not (done or truncated):
        action = action_fn(obs, env)
        obs, _, done, truncated, info = env.step(action)
        equity.append(info["net_worth"])
        positions.append(info["position"])
    result = summarize(env, info)
    result["equity"] = equity
    result["positions"] = positions
    return result


def position_mix(positions):
    arr = np.asarray(positions, dtype=float)
    return {
        "cash_0": float(np.mean(arr < 0.25) * 100),
        "half_50": float(np.mean((arr >= 0.25) & (arr < 0.75)) * 100),
        "full_100": float(np.mean(arr >= 0.75) * 100),
    }


def print_table(rows):
    print("\nEstrategia              | Retorno   | Máx. drawdown | Costos")
    print("------------------------+-----------+---------------+--------------")
    for name, metrics in rows:
        print(
            f"{name:<23} | {metrics['return_pct']:>7.2f}% "
            f"| {metrics['max_drawdown_pct']:>11.2f}% "
            f"| ${metrics['costs']:>10.2f}"
        )
    print()
