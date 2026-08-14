import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_family(family, source):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / family)
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


class CheckpointDirectoryTests(unittest.TestCase):
    def test_network_save_creates_missing_parent_directories(self):
        cases = {
            "DQN": """
                from pathlib import Path
                from tempfile import TemporaryDirectory
                from common import NNBase

                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'nested' / 'params' / 'model.pt'
                    NNBase().save(path)
                    assert path.is_file()
            """,
            "DDPG": """
                from pathlib import Path
                from tempfile import TemporaryDirectory
                from common import NetworkBase

                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'nested' / 'params' / 'model.pt'
                    NetworkBase().save(path)
                    assert path.is_file()
            """,
            "TD3": """
                from pathlib import Path
                from tempfile import TemporaryDirectory
                from common import NetworkBase

                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'nested' / 'params' / 'model.pt'
                    NetworkBase().save(path)
                    assert path.is_file()
            """,
            "SAC": """
                from pathlib import Path
                from tempfile import TemporaryDirectory
                from sac_discrete import DiscreteSAC

                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'nested' / 'params' / 'model.pt'
                    network = DiscreteSAC(
                        1e-3, 1e-3, 4, 16, 2, device='cpu'
                    )
                    network.save(path)
                    assert path.is_file()
            """,
        }

        for family, source in cases.items():
            with self.subTest(family=family):
                run_family(family, source)


if __name__ == "__main__":
    unittest.main()
