"""Command-line interface for hvacbench demos and smoke tests."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from hvacbench import __version__
from hvacbench.boptest.bestest_air import BestestAir
from hvacbench.config import EnvConfig
from hvacbench.energy_price import EnergyPriceType
from hvacbench.envs import BoptestEvaluationEnv, BoptestRolloutEnv, SafeEnv, TTMEnv
from hvacbench.envs.base import BaseEnv
from hvacbench.models.mock import MockTTM
from hvacbench.providers.mock import MockProvider
from hvacbench.rewards.simple import SimpleReward
from hvacbench.safety.control_safety import ControlSafetyFilter
from hvacbench.schemas import FloatArray


STEP_PERIOD_SECONDS = 900


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _control_plan(config: EnvConfig, heating: float, cooling: float) -> FloatArray:
    return np.tile(
        np.array([[heating, cooling]], dtype=np.float64),
        (config.horizon, 1),
    )


def _simulation_config(args: argparse.Namespace) -> EnvConfig:
    return EnvConfig(
        history_length=args.history_length,
        horizon=args.horizon,
        total_simulation_seconds=args.steps * STEP_PERIOD_SECONDS,
    )


def _state_from_info(info: dict[str, Any]) -> FloatArray:
    states = np.asarray(info["predicted_states"], dtype=np.float64)
    return states[0]


def _print_run_header(name: str, config: EnvConfig, args: argparse.Namespace) -> None:
    print(f"hvacbench {__version__} | {name}")
    print(
        "history_length="
        f"{config.history_length} horizon={config.horizon} steps={args.steps}"
    )
    print("step reward temp_c power_w heat_sp cool_sp truncated")


def _print_step(
    step: int,
    reward: float,
    state: FloatArray,
    applied_control: FloatArray,
    truncated: bool,
) -> None:
    print(
        f"{step:>4} "
        f"{reward:>7.2f} "
        f"{state[0]:>6.2f} "
        f"{state[1]:>7.2f} "
        f"{applied_control[0]:>7.2f} "
        f"{applied_control[1]:>7.2f} "
        f"{str(truncated).lower()}"
    )


def _close_if_supported(env: BaseEnv) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _run_env(name: str, env: BaseEnv, config: EnvConfig, args: argparse.Namespace) -> int:
    control_plan = _control_plan(
        config,
        heating=args.heating_setpoint,
        cooling=args.cooling_setpoint,
    )
    _print_run_header(name, config, args)

    try:
        for step in range(1, args.steps + 1):
            _obs, reward, terminated, truncated, info = env.step(control_plan)
            _print_step(
                step=step,
                reward=reward,
                state=_state_from_info(info),
                applied_control=np.asarray(info["applied_control"], dtype=np.float64),
                truncated=truncated,
            )
            if terminated or truncated:
                break
    finally:
        _close_if_supported(env)

    return 0


def _add_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=_positive_int, default=10)
    parser.add_argument("--history-length", type=_positive_int, default=8)
    parser.add_argument("--horizon", type=_positive_int, default=8)
    parser.add_argument("--heating-setpoint", type=float, default=22.0)
    parser.add_argument("--cooling-setpoint", type=float, default=24.0)


def _add_energy_price_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--energy-price",
        choices=[price.value for price in EnergyPriceType],
        default=EnergyPriceType.DYNAMIC.value,
        help=(
            "Electricity price profile used by the provider or BOPTEST scenario "
            "(default: dynamic)."
        ),
    )


def _add_boptest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1")
    parser.add_argument("--start-day", type=_non_negative_int, default=0)
    _add_energy_price_arg(parser)


def _cmd_info(_args: argparse.Namespace) -> int:
    config = EnvConfig()
    print(f"hvacbench {__version__}")
    print("purpose: validate learned HVAC surrogates for receding-horizon control")
    print("default testcase: BOPTEST bestest_air")
    print("energy price modes: constant, dynamic, highly_dynamic")
    print(
        "default env config: "
        f"history_length={config.history_length}, horizon={config.horizon}, "
        f"step_period_seconds={config.step_period_seconds}"
    )
    print("environments: TTMEnv, BoptestRolloutEnv, BoptestEvaluationEnv")
    print("offline demo: hvacbench mock-rollout --steps 5")
    print("TTM demo: hvacbench ttm-rollout --model-path gft/ttm4hvac")
    return 0


def _cmd_mock_rollout(args: argparse.Namespace) -> int:
    config = _simulation_config(args)
    provider = MockProvider(config=config, seed=args.seed)
    model = MockTTM(config=config)
    reward = SimpleReward(config=config)
    base_env = TTMEnv(
        config=config,
        provider=provider,
        reward=reward,
        model=model,
    )
    env: BaseEnv
    if args.no_safety:
        env = base_env
    else:
        env = SafeEnv(base_env, safety_filter=ControlSafetyFilter())
    return _run_env("mock learned-surrogate rollout", env, config, args)


def _cmd_ttm_rollout(args: argparse.Namespace) -> int:
    config = _simulation_config(args)
    reward = SimpleReward(config=config)
    env = TTMEnv(
        config=config,
        reward=reward,
        model_path=args.model_path,
        energy_price_type=EnergyPriceType(args.energy_price),
        start_day=args.start_day,
    )
    return _run_env(f"TTM learned-surrogate rollout ({args.model_path})", env, config, args)


def _boptest_testcase(args: argparse.Namespace) -> BestestAir:
    return BestestAir(
        base_url=args.base_url,
        energy_price_type=EnergyPriceType(args.energy_price),
    )


def _cmd_boptest_rollout(args: argparse.Namespace) -> int:
    config = _simulation_config(args)
    reward = SimpleReward(config=config)
    env = BoptestRolloutEnv(
        config=config,
        reward=reward,
        testcase=_boptest_testcase(args),
        start_day=args.start_day,
    )
    return _run_env("BOPTEST horizon rollout", env, config, args)


def _cmd_boptest_evaluate(args: argparse.Namespace) -> int:
    config = _simulation_config(args)
    reward = SimpleReward(config=config)
    env = BoptestEvaluationEnv(
        config=config,
        reward=reward,
        testcase=_boptest_testcase(args),
        start_day=args.start_day,
    )
    return _run_env("BOPTEST deployment-style evaluation", env, config, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hvacbench",
        description=(
            "Run hvacbench smoke tests and demos for validating learned HVAC "
            "surrogates."
        ),
    )
    parser.add_argument("--version", action="version", version=f"hvacbench {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Show package and backend info.")
    info_parser.set_defaults(func=_cmd_info)

    mock_parser = subparsers.add_parser(
        "mock-rollout",
        help="Run an offline mock learned-surrogate rollout.",
    )
    _add_rollout_args(mock_parser)
    mock_parser.add_argument("--seed", type=int, default=42)
    mock_parser.add_argument("--no-safety", action="store_true")
    mock_parser.set_defaults(func=_cmd_mock_rollout)

    ttm_parser = subparsers.add_parser(
        "ttm-rollout",
        help="Run a TTM-backed learned-surrogate rollout.",
    )
    _add_rollout_args(ttm_parser)
    ttm_parser.add_argument(
        "--model-path",
        default="gft/ttm4hvac",
        help="Local path or Hugging Face model id for a compatible TTM checkpoint.",
    )
    ttm_parser.add_argument("--start-day", type=_non_negative_int, default=0)
    _add_energy_price_arg(ttm_parser)
    ttm_parser.set_defaults(func=_cmd_ttm_rollout)

    rollout_parser = subparsers.add_parser(
        "boptest-rollout",
        help="Run a BOPTEST horizon rollout smoke test.",
    )
    _add_rollout_args(rollout_parser)
    _add_boptest_args(rollout_parser)
    rollout_parser.set_defaults(func=_cmd_boptest_rollout)

    evaluate_parser = subparsers.add_parser(
        "boptest-evaluate",
        help="Run a deployment-style BOPTEST evaluation smoke test.",
    )
    _add_rollout_args(evaluate_parser)
    _add_boptest_args(evaluate_parser)
    evaluate_parser.set_defaults(func=_cmd_boptest_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], int] = args.func
    return command(args)


if __name__ == "__main__":
    raise SystemExit(main())
