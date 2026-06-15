import os
MET='results/aco_results/checkpoints'
methods=['fixed_time','hybrid_dqn','hybrid_ddqn','hybrid_ppo']
seeds=[101,202,303,404,505]
missing=[]
existing=[]
for m in methods:
    for s in seeds:
        if m=='fixed_time':
            name=f'fixed_time_seed{s}_metrics_latest.json'
        else:
            agent=m.replace('hybrid_','')
            name=f'hybrid_{agent}_seed{s}_metrics_latest.json'
        path=os.path.join(MET,name)
        if os.path.exists(path):
            existing.append(path)
        else:
            missing.append(path)
print('existing_count',len(existing))
print('missing_count',len(missing))
for p in missing:
    print('MISSING',p)
