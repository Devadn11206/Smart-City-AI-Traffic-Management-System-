import torch, os, json

paths=[
 'results/checkpoint_backups/20260530_203534/hybrid_dqn_model_best.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_dqn_model_final.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_dqn_model.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_ddqn_model_best.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_ddqn_model_final.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_ddqn_model.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_ppo_model_best.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_ppo_model_final.pth',
 'results/checkpoint_backups/20260530_203534/hybrid_ppo_model.pth',
]

out = []
for p in paths:
    entry = {'path': p}
    if not os.path.exists(p):
        entry['status']='missing'
        out.append(entry)
        continue
    try:
        ck = torch.load(p, map_location='cpu')
        entry['status']='loaded'
        # metadata
        meta = {}
        if isinstance(ck, dict):
            for k in ('agent','state_dim','action_dim','version'):
                if k in ck:
                    meta[k]=ck[k]
        entry['metadata']=meta
        # extract state_dict
        sd = None
        if isinstance(ck, dict) and 'state_dict' in ck:
            sd = ck['state_dict']
        elif isinstance(ck, dict) and any(isinstance(v, torch.Tensor) for v in ck.values()):
            sd = {k:v for k,v in ck.items() if isinstance(v, torch.Tensor)}
        else:
            sd = ck
        # first few param shapes
        params = {}
        for k,v in sd.items():
            params[k]=tuple(v.shape)
        entry['param_shapes'] = {k:params[k] for k in list(params)[:10]}
    except Exception as e:
        entry['status']='error'
        entry['error']=str(e)
    out.append(entry)

print(json.dumps(out, indent=2))
