"""Collect audited MINT/WiLoR raw EEF evaluation into CSV and Markdown."""
import argparse
import csv
import json
from pathlib import Path

METRICS = ['rho_pos', 'rho_rot', 'rho_se3', 'D_pos_mm', 'D_rot_deg']


def collect(root, tasks):
    rows, coverage = [], []
    for task in tasks:
        directory = root / f'mint_raw_{task}'
        manifest = json.loads((directory / 'manifest.json').read_text())
        health = json.loads((directory / 'health.json').read_text())
        for method in ['wilor_raw', 'mint_raw']:
            for side in manifest['active_sides']:
                frames = valid = 0
                for record in manifest['variants'][method]['episodes'].values():
                    eef = json.loads(Path(record['final_eef']).read_text())
                    frames += len(eef['frames'])
                    valid += sum(f.get('hand_l' if side == 'left' else 'hand_r') is not None for f in eef['frames'])
                coverage.append(dict(task=task, method=method, side=side, episodes=len(health),
                                     healthy_episodes=sum(item[method][side][0] is None for item in health.values()),
                                     valid_frames=valid, total_frames=frames))
        for cohort in ['all', 'common']:
            comparison_path = directory / cohort / 'comparison.json'
            if not comparison_path.exists():
                selection = json.loads((directory / 'cohort.json').read_text())
                if ((directory / cohort / 'unavailable.json').exists()
                        or (cohort == 'common' and not selection['common_healthy'])):
                    continue
                raise FileNotFoundError(f'Evaluation is incomplete: {comparison_path}')
            comparison = json.loads(comparison_path.read_text())
            if cohort == 'common' and not comparison['comparable']:
                raise ValueError(f'{task}: common cohort still has unequal evaluated pairs; inspect exclusions')
            for method in ['wilor_raw', 'mint_raw']:
                summary = json.loads((directory / cohort / method / 'summary.json').read_text())
                per_side = summary['metrics']['per_side']
                for side, block in per_side.items():
                    rows.append(dict(task=task, cohort=cohort, method=method, side=side,
                                     n_ego=block['n_ego_episodes'], comparable=comparison['comparable'],
                                     **{k: block['aggregate'][k]['mean'] for k in METRICS}))
                rows.append(dict(task=task, cohort=cohort, method=method, side='active_side_mean',
                                 n_ego=min(v['n_ego_episodes'] for v in per_side.values()),
                                 comparable=comparison['comparable'],
                                 **{k: summary['metrics']['bimanual_active_side_mean'][k] for k in METRICS}))
    return rows, coverage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs'))
    p.add_argument('--output', type=Path, default=Path('outputs/mint_raw_comparison'))
    p.add_argument('--tasks', nargs='+', default=['redcube', 'cola'])
    a = p.parse_args()
    rows, coverage = collect(a.root, a.tasks)
    a.output.mkdir(parents=True, exist_ok=True)
    for filename, records in [('metrics.csv', rows), ('coverage.csv', coverage)]:
        if records:
            with (a.output / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    (a.output / 'metrics.json').write_text(json.dumps(dict(metrics=rows, coverage=coverage), indent=2) + '\n')
    lines = ['# MINT / WiLoR 校正前 EEF 评估', '',
             '以下距离表仅使用两方法所有活动臂均健康的共同 episode 集合；指标越低越好。', '',
             '|任务|方法|共同 episodes|rho_pos|rho_rot|rho_se3|D_pos_mm|D_rot_deg|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['cohort'] == 'common' and row['side'] == 'active_side_mean':
            lines.append('|' + '|'.join([row['task'], row['method'], str(row['n_ego'])] +
                                       [f'{row[k]:.3f}' for k in METRICS]) + '|')
    lines += ['', '## 全部输入的有效率', '', '|任务|方法|臂|事件健康 episodes|有效 EEF 帧|', '|---|---|---|---:|---:|']
    for row in coverage:
        lines.append(f"|{row['task']}|{row['method']}|{row['side']}|{row['healthy_episodes']}/{row['episodes']}|{row['valid_frames']}/{row['total_frames']}|")
    lines += ['', '完整 all/common 分臂指标见 metrics.csv。all 的两方法有效集合可能不同，不能直接将其差值解释为方法提升。',
              '共同集合指标也具有筛选效应，必须连同有效率解释。这是相对于留出真机的轨迹分布距离，不是配对重建真值误差。', '']
    (a.output / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
