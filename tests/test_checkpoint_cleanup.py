import importlib.util
import os
import tempfile
import unittest


MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ddqn_pinn_package',
    'training',
    'train_pi_ddqn_10x10.py',
)


spec = importlib.util.spec_from_file_location('train_pi_ddqn_10x10', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class _DummyModel:
    def state_dict(self):
        return {'weight': 1.0}


class _DummyAgent:
    def __init__(self):
        self.model = _DummyModel()
        self.target_model = _DummyModel()
        self.epsilon = 0.1


class CheckpointCleanupTest(unittest.TestCase):
    def test_save_checkpoint_keeps_latest_file_and_prunes_oldest_by_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            module.CHECKPOINT_DIR = temp_dir
            os.makedirs(temp_dir, exist_ok=True)

            for ep in (2, 35, 100):
                path = os.path.join(temp_dir, f'ckpt_ep{ep}.pt')
                with open(path, 'wb') as fh:
                    fh.write(b'placeholder')

            agents = {'ts1': _DummyAgent()}
            pinn_ensemble = type('Pinn', (), {'state_dict': lambda self: {'pinn': 2.0}})()

            module.save_checkpoint(agents, pinn_ensemble, ['ts1'], 100, {'ts1': 0.1})

            remaining = sorted(os.listdir(temp_dir))
            self.assertEqual(remaining, ['ckpt_ep100.pt', 'ckpt_ep35.pt'])
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'ckpt_ep100.pt')))


if __name__ == '__main__':
    unittest.main()
