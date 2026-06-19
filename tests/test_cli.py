import pytest

from hvacbench.cli import build_parser, main


def test_cli_info(capsys):
    exit_code = main(["info"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "hvacbench" in captured.out
    assert "TTMEnv" in captured.out
    assert "BoptestRolloutEnv" in captured.out


def test_cli_mock_rollout(capsys):
    exit_code = main(
        [
            "mock-rollout",
            "--steps",
            "2",
            "--history-length",
            "4",
            "--horizon",
            "4",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "mock learned-surrogate rollout" in captured.out
    assert "step reward temp_c power_w heat_sp cool_sp truncated" in captured.out
    assert "   1 " in captured.out
    assert "   2 " in captured.out


def test_cli_ttm_rollout_parses_model_path():
    parser = build_parser()
    args = parser.parse_args(
        [
            "ttm-rollout",
            "--model-path",
            "local-model",
            "--energy-price",
            "highly_dynamic",
            "--start-day",
            "12",
        ]
    )

    assert args.model_path == "local-model"
    assert args.energy_price == "highly_dynamic"
    assert args.start_day == 12


def test_cli_ttm_rollout_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["ttm-rollout", "--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--model-path" in captured.out
    assert "--energy-price" in captured.out
