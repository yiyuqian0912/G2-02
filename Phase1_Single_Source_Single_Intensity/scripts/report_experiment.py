"""Render completed frozen experiments; never selects or trains a model."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/experiments/round1_cpu'))
    args=parser.parse_args()
    root=args.root
    protocol=read(root/'protocol.json')
    c=protocol['config']
    seeds=c['training']['seeds']
    variants=[v for v in c['variants'] if (root/v/'summary.json').exists()]
    if not variants:
        raise ValueError('no completed experiment summaries')
    reports={v:[read(root/v/f'seed_{seed}'/'test_metrics.json') for seed in seeds] for v in variants}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(variants),len(seeds),figsize=(4*len(seeds),3*len(variants)),squeeze=False,layout='constrained')
    for row,v in enumerate(variants):
        for col,seed in enumerate(seeds):
            h=read(root/v/f'seed_{seed}'/'training.json')['history']
            ax=axes[row,col]
            ax.plot([r['epoch'] for r in h],[r['train_ce'] for r in h],label='Train')
            ax.plot([r['epoch'] for r in h],[r['validation_ce'] for r in h],label='Validation')
            ax.axvline(reports[v][col]['best_epoch'],color='gray',ls=':',label='Selected epoch')
            ax.set(title=f'{v}, seed {seed}',xlabel='Epoch',ylabel='Cross entropy')
            ax.legend(fontsize=8)
    fig.savefig(root/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    keys=['forward_kl','student_compatible_mass']
    names=['Forward KL to own teacher (lower is better)','Compatible probability mass (higher is better)']
    sizes=c['window_sizes']
    for ax,key,title in zip(axes,keys,names):
        for v in variants:
            values=np.array([[r['by_window_size'][str(size)]['metrics'][key]['scene_mean'] for size in sizes] for r in reports[v]])
            ax.errorbar(range(len(sizes)),values.mean(0),yerr=values.std(0,ddof=1) if len(seeds)>1 else np.zeros(len(sizes)),marker='o',capsize=3,label=v)
            baseline=[reports[v][0]['uniform_baseline_by_window_size'][str(size)]['metrics'][key]['scene_mean'] for size in sizes]
            ax.plot(range(len(sizes)),baseline,ls='--',alpha=.6,label=f'{v} uniform baseline')
        ax.set(title=title,xlabel='Window size',xticks=range(len(sizes)),xticklabels=sizes)
        ax.legend(fontsize=8)
    fig.savefig(root/'window_metrics.png',dpi=180);plt.close(fig)
    cost_key = 'student_expected_penalized_cost' if all('student_expected_penalized_cost' in r['aggregate']['metrics'] for rr in reports.values() for r in rr) else 'student_expected_physical_cost'
    lines=['# Phase I 第一轮实验结果','',
        f'固定 {sum(c["scene_counts"].values())} 个场景：训练 {c["scene_counts"]["train"]}、验证 {c["scene_counts"]["validation"]}、测试 {c["scene_counts"]["test"]}。',
        f'窗口 size={sizes}；每场景每种 size 选 {c["views_per_size"]} 个；候选 spacing={c["teacher"]["spacing"]}；训练 seeds={seeds}。','',
        '每个模型按验证集最低 cross-entropy 选择。下表为测试集的场景平均，再报告训练种子间均值 ± 标准差；不是跨数据集置信区间。','',
        '| 方法 | KL↓ | L1↓ | 兼容概率质量↑ | '+('有限惩罚代价↓' if cost_key.endswith('penalized_cost') else '期望物理代价↓')+' |',
        '|---|---:|---:|---:|---:|']
    metric_keys=['forward_kl','l1_distance','student_compatible_mass',cost_key]
    for v in variants:
        for label,field in [(v,'aggregate'),(v+'（共同 response 参考）','common_response_reference')]:
            if v=='response' and field=='common_response_reference':continue
            cells=[]
            for key in metric_keys:
                values=[r[field]['metrics'][key]['scene_mean'] for r in reports[v]]
                cells.append(f'{np.mean(values):.6f} ± {np.std(values,ddof=1) if len(values)>1 else 0:.6f}')
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        values=[reports[v][0]['uniform_baseline']['metrics'][key]['scene_mean'] for key in metric_keys]
        lines.append('| '+v+' 对应的 uniform baseline | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines+=['','response 与 edge 的自身物理代价定义不同，不直接比较原始代价值。评估边界项是否有增益时，使用 edge 的“共同 response 参考”与 response 对比。',
        '', '## 训练与网格覆盖','', '| 方法 | 最优 epoch（按种子顺序） | 全部观测中无兼容网格点 | 测试集中无兼容网格点 |','|---|---|---:|---:|']
    for v in variants:
        rows=read(root/v/'teachers.json')['records']
        gaps=sum(r['compatible_candidates']==0 for r in rows)
        test_gaps=sum(r['compatible_candidates']==0 and r['split']=='test' for r in rows)
        lines.append(f'| {v} | {[r["best_epoch"] for r in reports[v]]} | {gaps}/{len(rows)} | {test_gaps}/{len(reports[v][0]["observations"])} |')
    lines+=['','无兼容网格点的观测仍保留在评估中。此时 compatible mass=0 不能单独归因于学生；需要检查候选网格分辨率。',
        '', '## 搜索对照','']
    for v in variants:
        path=root/v/'search_comparison.json'
        if path.exists():
            search=read(path)
            complete=[r for r in search['observations'] if r['complete']]
            lines.append(f'- {v}：{len(complete)}/{len(search["observations"])} 个 adaptive 搜索完成。细节见 [{v}/search_comparison.json]({v}/search_comparison.json)。')
    lines+=['','## 解读限制','',
        f'- 输入局部障碍掩码：{c["training"].get("use_obstacle", True)}；归一化范围：{c["training"].get("normalization_support", "geometry")}。world 使用公开范围；geometry 使用隐藏几何排除障碍物。',
        '- 有限惩罚代价以 alpha+2beta 惩罚障碍物内概率质量；不是含无穷无效点的真实期望物理代价。',
        '- 三个种子共享同一场景划分，标准差仅衡量训练随机性。',
        '- 本轮未覆盖 size 256/512、全量 10,000 场景、细网格收敛或一条/两条独立边界的配对验证。',
        '- 隐藏场景几何不作为学生输入，局部信息可能不足以恢复每个场景的精确教师分布。',
        '', '## 图与原始记录','',
        '![训练曲线](learning_curves.png)','',
        '![按窗口大小比较](window_metrics.png)','',
        '- [冻结协议](protocol.json)',
        *[f'- [{v} 汇总]({v}/summary.json)，逐种子报告和预测保存在 `{v}/seed_<seed>/`。' for v in variants]]
    (root/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    print(root/'RESULTS.zh-CN.md')


if __name__=='__main__':main()
