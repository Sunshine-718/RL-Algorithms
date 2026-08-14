import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path):
    path = ROOT / relative_path
    for name in ("common", "replaybuffer"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "temperature_" + relative_path.replace("/", "_").replace(".", "_"),
            path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class SACTemperatureTests(unittest.TestCase):
    def test_discrete_temperature_gradient_follows_entropy_error(self):
        module = load_module("SAC/sac_discrete.py")
        target_entropy = math.log(4) * 0.8

        def gradient(probabilities, initial_alpha):
            log_alpha = torch.nn.Parameter(
                torch.tensor([[math.log(initial_alpha)]])
            )
            probabilities = torch.tensor([probabilities])
            loss = module.discrete_temperature_loss(
                log_alpha,
                probabilities.log(),
                probabilities,
                target_entropy,
            )
            loss.backward()
            return float(log_alpha.grad.item())

        uniform_gradient = gradient([0.25, 0.25, 0.25, 0.25], 0.2)
        low_entropy_gradient = gradient([0.97, 0.01, 0.01, 0.01], 0.2)
        large_alpha_gradient = gradient([0.97, 0.01, 0.01, 0.01], 200.0)

        self.assertGreater(uniform_gradient, 0)
        self.assertLess(low_entropy_gradient, 0)
        self.assertAlmostEqual(low_entropy_gradient, large_alpha_gradient, places=6)

    def test_continuous_temperature_gradient_follows_entropy_error(self):
        module = load_module("SAC/sac_continuous.py")

        def gradient(log_prob):
            log_alpha = torch.nn.Parameter(torch.tensor([[math.log(0.2)]]))
            loss = module.continuous_temperature_loss(
                log_alpha,
                torch.tensor([[log_prob]]),
                target_entropy=-2,
            )
            loss.backward()
            return float(log_alpha.grad.item())

        self.assertGreater(gradient(0.0), 0)
        self.assertLess(gradient(3.0), 0)

    def test_all_sac_agents_report_uncapped_temperature(self):
        cases = [
            ("SAC/sac_continuous.py", "ContinuousSACAgent"),
            ("SAC/qrsac_continuous.py", "ContinuousSACAgent"),
            ("SAC/sac_discrete.py", "DiscreteSACAgent"),
            ("SAC/qrsac_discrete.py", "DiscreteSACAgent"),
            ("SAC/qrsac_lunarlander.py", "DiscreteSACAgent"),
        ]
        for relative_path, class_name in cases:
            with self.subTest(path=relative_path):
                module = load_module(relative_path)
                agent = object.__new__(getattr(module, class_name))
                agent.net = SimpleNamespace(
                    alpha=torch.nn.Parameter(torch.tensor([[math.log(2.5)]]))
                )
                self.assertAlmostEqual(agent.alpha, 2.5, places=6)

    def test_legacy_checkpoint_resets_temperature_but_keeps_weights(self):
        module = load_module("SAC/sac_discrete.py")
        legacy = module.DiscreteSAC(
            1e-3, 1e-3, 4, 16, 2, alpha=0.2, device="cpu"
        )
        with torch.no_grad():
            legacy.alpha.fill_(9.65)
            next(legacy.pi.parameters()).fill_(0.125)

        payload = {
            "model": legacy.state_dict(),
            "actor_opt": legacy.actor_opt.state_dict(),
            "critic_opt": legacy.critic_opt.state_dict(),
            "alpha_opt": legacy.alpha_opt.state_dict(),
            "alpha": legacy.alpha.detach().clone(),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            restored = module.DiscreteSAC(
                1e-3, 1e-3, 4, 16, 2, alpha=0.2, device="cpu"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                restored.load(path)

        self.assertAlmostEqual(
            float(restored.alpha.exp().item()), 0.2, places=6
        )
        torch.testing.assert_close(
            next(restored.pi.parameters()),
            next(legacy.pi.parameters()),
        )

    def test_current_checkpoint_preserves_temperature_above_one(self):
        module = load_module("SAC/sac_discrete.py")
        network = module.DiscreteSAC(
            1e-3, 1e-3, 4, 16, 2, alpha=0.2, device="cpu"
        )
        with torch.no_grad():
            network.alpha.fill_(math.log(2.5))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.pt"
            network.save(path)
            restored = module.DiscreteSAC(
                1e-3, 1e-3, 4, 16, 2, alpha=0.2, device="cpu"
            )
            restored.load(path)

        self.assertAlmostEqual(
            float(restored.alpha.exp().item()), 2.5, places=6
        )


if __name__ == "__main__":
    unittest.main()
