import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
import torch
from pinn_models.pinn_model import TrafficPINN
from utils.reliability import extract_state_dict, load_torch_checkpoint

MODEL_PATH = os.path.join(ROOT, 'pinn_models', 'trained_pinn.pth')
print('PINN model path:', MODEL_PATH)

m = TrafficPINN()
if os.path.exists(MODEL_PATH):
    try:
        ckpt = load_torch_checkpoint(MODEL_PATH, map_location='cpu')
        m.load_state_dict(extract_state_dict(ckpt))
        print('Loaded PINN weights from checkpoint')
    except Exception as e:
        print('Failed to load PINN:', e)
else:
    print('PINN weights not found, using random init')

m.eval()
with torch.no_grad():
    t = torch.zeros((5,1))
    x = torch.linspace(0,1,5).unsqueeze(1)
    rho, v = m(t, x)
    print('rho:', rho.squeeze().tolist())
    print('v:', v.squeeze().tolist())
print('PINN smoke test done')
