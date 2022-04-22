import matplotlib.pyplot as plt
import os
import os.path as osp
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple, Sequence, Union
from pathlib import Path
from collections import OrderedDict, defaultdict
import argparse
import collections
import functools
import multiprocessing as mp
import pathlib
import re
import subprocess

import pandas as pd
Run = collections.namedtuple('Run', 'task method seed xs ys color')


COLORS = {
    'Our impl.': 'tab:red',
    'Our impl.-TF init': 'tab:orange',
    'Dreamer': 'tab:blue',
    'Dreamer-TF2': 'tab:green',
}

def plot(
    labeled_data: Dict[str, Dict[str, Sequence]],
    ax: Optional[plt.Axes] = None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    xlim: Tuple[Optional[float], Optional[float]] = (None, None),
    ylim: Tuple[Optional[float], Optional[float]] = (None, None),
    linewidth: float = 2,
    alpha: float = 0.1,
    legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        assert figsize is None

    for label, line_data in labeled_data.items():
        x = np.array(line_data['x'])
        y = np.array(line_data['y'])
        assert len(x) == len(y) and x.ndim == y.ndim == 1

        if label in COLORS:
            color = COLORS[label]
        else:
            color = None
        line = ax.plot(x, y, label=label, linewidth=linewidth, color=color)[0]
        if 'y_std' in line_data:
            y_std = np.array(line_data['y_std'])
            assert len(y_std) == len(y) and y_std.ndim == y.ndim == 1
            color = line.get_color()
            ax.fill_between(x, y - y_std, y + y_std, alpha=alpha, color=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        ax.legend()
    return ax

def plot_grid(
    titled_data: Dict[str, Dict[str, Dict[str, Sequence]]],
    ncols: Optional[int] = None,
    xlabel: str = '',
    ylabel: str = '',
    xlim: Tuple[Optional[float], Optional[float]] = (None, None),
    ylim: Tuple[Optional[float], Optional[float]] = (None, None),
    linewidth: float = 2,
    alpha: float = 0.1,
    legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    if ncols is None:
        ncols = len(titled_data)
        nrows = 1
    else:
        nrows = len(titled_data) // ncols + int(len(titled_data) % ncols != 0)

    # Caculate figure size
    if figsize is None:
        # Figure size is in inches (2.54 cm). This is the default value.
        width = ncols * 6.4
        height = nrows * 4.8
        figsize = (width, height)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, (title, labeled_data) in zip(axes.ravel(), titled_data.items()):
        plot(labeled_data, ax=ax, title=title, xlabel=xlabel, ylabel=ylabel,
             xlim=xlim, ylim=ylim, linewidth=linewidth, alpha=alpha,
             legend=legend)
        # Only put legend on first figure
        legend = False

    return fig, axes

def setup_style():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    LARGE_SIZE = 20

    # General configuration: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
    # Font: https://matplotlib.org/stable/tutorials/text/text_props.html#default-font
    default_font = {
        'family': ['sans-serif'],  #  or serif.
        'sans-serif': ['Dejavu Sans', 'Computer Modern Sans Serif', 'Helvetica',  'sans-serif'],  # first one will be used. Dejavu is just the default
        'serif': ['Computer Modern Roman', 'Times New Roman', 'serif'],
        'weight': 'normal',  # or light, bold
        'size': SMALL_SIZE
    }

    # Style-sheets: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    plt.style.use('bmh')
    # You don't need latex: https://matplotlib.org/3.5.0/tutorials/text/mathtext.html
    # plt.rc('text', usetex=True)
    plt.rc('font', **default_font)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    # Color cycle
    # Specifying colors: https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    # Specifying line styles: https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/linestyles.html
    # Cycler: https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
    from cycler import cycler
    # prop_cycle = cycler(color=['r', 'g', 'b', 'y']) + cycler(linestyle=['-', '--', ':', '-.'])
    # prop_cycle = cycler(color=['tab:red', 'tab:green', 'tab:blue']) * cycler(linestyle=['-', '--', ':', '-.'])
    prop_cycle = plt.rcParams['axes.prop_cycle']  # The default one
    plt.rc('axes', prop_cycle=prop_cycle)


def load_data(data_dir: Path, pattern: str, keys: List[str]):
    data = defaultdict(list)
    matched = list(data_dir.glob(pattern))
    assert len(matched) == 1, f'Expected only 1 match for "{pattern}"\nbut there are {len(matched)}.'
    matched = matched[0]
    with matched.open('r') as f:
        for line in f:
            step_data = json.loads(line)
            for key in keys:
                data[key].append(step_data[key])
    return {k: np.array(v) for k, v in data.items()}

def load_our_data():
    logdir = Path('results/baselines/')
    tasks = ['cartpole_swingup', 'cartpole_balance', 'cup_catch', 'cheetah_run', 'hopper_stand', 'pendulum_swingup', 'reacher_easy', 'walker_stand', 'walker_walk']
    task2title = OrderedDict.fromkeys(tasks)
    for key in task2title:
        task2title[key] = key.replace('_', ' ').title()
    seeds = [0, 1, 2]

    titled_data = OrderedDict.fromkeys(task2title.values())
    for task in tasks:
        labeled_data = {'Our impl.': dict.fromkeys(['x', 'y', 'y_std'])}
        data_seeds = []
        for seed in seeds:
            pattern = f'*{task}*seed={seed}*/test_metrics.jsonl'
            data_seed = load_data(logdir, pattern, keys=['step', 'test/return'])
            data_seeds.append(data_seed)
        # Checks
        for data in data_seeds:
            assert (data['step'] == data_seeds[0]['step']).all()
            assert len(data['step']) == len(data['test/return'])

        labeled_data['Our impl.']['x'] = data_seeds[0]['step']
        labeled_data['Our impl.']['y'] = np.mean([data['test/return'] for data in data_seeds], axis=0)
        labeled_data['Our impl.']['y_std'] = np.std([data['test/return'] for data in data_seeds], axis=0)

        titled_data[task2title[task]] = labeled_data

    return titled_data

def load_dreamer_data():
    tasks = ['cartpole_swingup', 'cartpole_balance', 'cup_catch', 'cheetah_run', 'hopper_stand', 'pendulum_swingup', 'reacher_easy', 'walker_stand', 'walker_walk']
    task2title = OrderedDict.fromkeys(tasks)
    for key in task2title:
        task2title[key] = key.replace('_', ' ').title()

    titled_data = OrderedDict.fromkeys(task2title.values())
    with Path('scores/dreamer.json').open('r') as f:
        all_runs = json.load(f)
        for task in tasks:
            labeled_data = {'Dreamer': dict.fromkeys(['x', 'y', 'y_std'])}
            data_seeds = []
            min_len = np.inf
            for run in all_runs:
                if run['task'] == f'dmc_{task}':
                    data_seeds.append(dict(xs=np.array(run['xs']), ys=np.array(run['ys'])))
                min_len = min(min_len, len(run['xs']))
            for data in data_seeds:
                data['xs'] = data['xs'][:min_len]
                data['ys'] = data['ys'][:min_len]
                try:
                    assert (data['xs'] == data_seeds[0]['xs']).all()
                    assert len(data['xs']) == len(data['ys'])
                except:
                    import ipdb; ipdb.set_trace()
            # if 'reacher' in task:
                # import ipdb; ipdb.set_trace()
            labeled_data['Dreamer']['x'] = data_seeds[0]['xs']
            labeled_data['Dreamer']['y'] = np.mean([data['ys'] for data in data_seeds], axis=0)
            labeled_data['Dreamer']['y_std'] = np.std([data['ys'] for data in data_seeds], axis=0)
            titled_data[task2title[task]] = labeled_data
    return titled_data


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')

def load_dreamer_tf2_data():
    tasks = ['dmc_cup_catch', 'dmc_hopper_stand', 'dmc_pendulum_swingup']
    args = AttrDict()
    args.indir = [Path('results/dreamer_tf2/')]
    args.xaxis = 'step'
    args.yaxis = 'test/return'
    args.palette = PALETTE
    args.bins = 3e4
    args.tasks = tasks
    args.colors = {}
    args.methods = ['dreamer']
    args.tasks = [re.compile(p) for p in args.tasks]
    args.methods = [re.compile(p) for p in args.methods]
    runs = load_runs(args)

    titled_data = defaultdict(dict)
    for task in tasks:
        relevant = [r for r in runs if r.task == task]
        x, y, y_std = curve_std(relevant, args)
        title = task.replace('_', ' ').title()[4:]
        titled_data[title]['Dreamer-TF2'] = {}
        titled_data[title]['Dreamer-TF2']['x'] = x
        titled_data[title]['Dreamer-TF2']['y'] = y
        titled_data[title]['Dreamer-TF2']['y_std'] = y_std

    return titled_data

def load_run(filename, indir, args):
  task, method, seed = filename.relative_to(indir).parts[:-1]
  try:
    # Future pandas releases will support JSON files with NaN values.
    # df = pd.read_json(filename, lines=True)
    with filename.open() as f:
      df = pd.DataFrame([json.loads(l) for l in f.readlines()])
  except ValueError as e:
    print('Invalid', filename.relative_to(indir), e)
    return
  try:
    df = df[[args.xaxis, args.yaxis]].dropna()
  except KeyError:
    return
  xs = df[args.xaxis].to_numpy()
  ys = df[args.yaxis].to_numpy()
  color = args.colors[method]
  return Run(task, method, seed, xs, ys, color)

def load_runs(args):
  toload = []
  for indir in args.indir:
    filenames = list(indir.glob('**/*.jsonl'))
    for filename in filenames:
      task, method, seed = filename.relative_to(indir).parts[:-1]
      if not any(p.search(task) for p in args.tasks):
        continue
      if not any(p.search(method) for p in args.methods):
        continue
      if method not in args.colors:
        args.colors[method] = args.palette[len(args.colors)]
      toload.append((filename, indir))
  print(f'Loading {len(toload)} of {len(filenames)} runs...')
  jobs = [functools.partial(load_run, f, i, args) for f, i in toload]
  # with mp.Pool(10) as pool:
    # promises = [pool.apply_async(j) for j in jobs]
    # runs = [p.get() for p in promises]
  runs = [j() for j in jobs]
  runs = [r for r in runs if r is not None]
  return runs

def binning(xs, ys, bins, reducer):
  binned_xs = np.arange(xs.min(), xs.max() + 1e-10, bins)
  binned_ys = []
  for start, stop in zip([-np.inf] + list(binned_xs), binned_xs):
    left = (xs <= start).sum()
    right = (xs <= stop).sum()
    binned_ys.append(reducer(ys[left:right]))
  binned_ys = np.array(binned_ys)
  return binned_xs, binned_ys

def curve_std(runs, args):
  if args.bins:
    for index, run in enumerate(runs):
      xs, ys = binning(run.xs, run.ys, args.bins, np.nanmean)
      runs[index] = run._replace(xs=xs, ys=ys)
  xs = np.concatenate([r.xs for r in runs])
  ys = np.concatenate([r.ys for r in runs])
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  if args.bins:
    reducer = lambda y: (np.nanmean(np.array(y)), np.nanstd(np.array(y)))
    xs, ys = binning(xs, ys, args.bins, reducer)
    ys, std = ys.T
  return xs, ys, std

def merge_titled_data(*titled_data_list):
    agg_data = defaultdict(dict)
    for titled_data in titled_data_list:
        for title, labeled_data in titled_data.items():
            for label, line_data in labeled_data.items():
                agg_data[title][label] = line_data
    return agg_data

def plot_results():

    # logdir = Path('/home/mila/z/zhixuan.lin/scratch/dreamer-pytorch/output_exps/baselines/')
    # titled_data = load_our_data()
    dreamer_data = load_dreamer_data()
    ours_data = load_our_data()
    titled_data = merge_titled_data(dreamer_data, ours_data)

    plot_grid(titled_data, ncols=3, xlabel='Frames', ylabel='Episode Return', xlim=(0, 1000000))
    plt.tight_layout()
    plt.savefig('figures/compare.png')
    # plt.show()

def plot_update_horizon():

    # logdir = Path('/home/mila/z/zhixuan.lin/scratch/dreamer-pytorch/output_exps/baselines/')
    # titled_data = load_our_data()
    # dreamer_data = load_dreamer_data()
    ours_data = load_our_data()
    new_data = OrderedDict()
    tasks = ['cheetah_run', 'reacher_easy', 'walker_walk']
    tasks = {t.replace('_', ' ').title() for t in tasks}
    for task, labeled_data in ours_data.items():
        if task in tasks:
            new_data[task] = {'update_horizon=15': labeled_data['Our impl.']}
    update_horizon_data = load_update_horizon()
    titled_data = merge_titled_data(update_horizon_data, new_data)

    plot_grid(titled_data, ncols=3, xlabel='Frames', ylabel='Episode Return', xlim=(0, 1000000))
    plt.tight_layout()
    plt.savefig('figures/update_horizon')
    # plt.show()

def plot_single_step_q():

    # logdir = Path('/home/mila/z/zhixuan.lin/scratch/dreamer-pytorch/output_exps/baselines/')
    # titled_data = load_our_data()
    # dreamer_data = load_dreamer_data()
    ours_data = load_our_data()
    new_data = OrderedDict()
    tasks = ['cheetah_run', 'reacher_easy', 'walker_walk']
    tasks = {t.replace('_', ' ').title() for t in tasks}
    for task, labeled_data in ours_data.items():
        if task in tasks:
            new_data[task] = {'multi-step Q': labeled_data['Our impl.']}
    update_horizon_data = load_single_step_q()
    titled_data = merge_titled_data(update_horizon_data, new_data)

    plot_grid(titled_data, ncols=3, xlabel='Frames', ylabel='Episode Return', xlim=(0, 1000000))
    plt.tight_layout()
    plt.savefig('figures/single_step_Q')
    # plt.show()

def plot_dreamer_tf2():

    # logdir = Path('/home/mila/z/zhixuan.lin/scratch/dreamer-pytorch/output_exps/baselines/')
    # titled_data = load_our_data()
    # dreamer_data = load_dreamer_data()
    ours_data = load_our_data()
    tasks = ['cup_catch', 'hopper_stand', 'pendulum_swingup']
    # tasks = ['pendulum_swingup']
    tasks = {t.replace('_', ' ').title() for t in tasks}
    dreamer_data = load_dreamer_data()
    dreamer_tf2_data = load_dreamer_tf2_data()
    ours_tf_init = load_tf_init()

    titled_data = merge_titled_data(dreamer_data, dreamer_tf2_data, ours_data, ours_tf_init)
    # titled_data = merge_titled_data(dreamer_data, dreamer_tf2_data)
    titled_data_new = {}
    for task, labeled_data in titled_data.items():
        if task in tasks:
            titled_data_new[task] = labeled_data
    plot_grid(titled_data_new, ncols=3, xlabel='Frames', ylabel='Episode Return', xlim=(0, 1000000))
    plt.tight_layout()
    plt.savefig('figures/dreamer_tf2.png')
    # plt.show()



def load_update_horizon():

    logdir = Path('results/update_horizon/')
    tasks = ['cheetah_run', 'reacher_easy', 'walker_walk']
    task2title = OrderedDict.fromkeys(tasks)
    for key in task2title:
        task2title[key] = key.replace('_', ' ').title()
    seeds = [0, 1, 2]
    update_horizons = [1, 5, 10]

    titled_data = OrderedDict.fromkeys(task2title.values())
    for task in tasks:
        labeled_data = {}
        for update_horizon in update_horizons:
            data_seeds = []
            for seed in seeds:
                pattern = f'*{task}*seed={seed},update_horizon={update_horizon}-*/test_metrics.jsonl'
                data_seed = load_data(logdir, pattern, keys=['step', 'test/return'])
                data_seeds.append(data_seed)
            # Checks
            for data in data_seeds:
                assert (data['step'] == data_seeds[0]['step']).all()
                assert len(data['step']) == len(data['test/return'])

            label = f'update_horizon={update_horizon}'
            labeled_data[label] = {}
            labeled_data[label]['x'] = data_seeds[0]['step']
            labeled_data[label]['y'] = np.mean([data['test/return'] for data in data_seeds], axis=0)
            labeled_data[label]['y_std'] = np.std([data['test/return'] for data in data_seeds], axis=0)
        titled_data[task2title[task]] = labeled_data

    return titled_data

def load_single_step_q():

    logdir = Path('results/single_step_q/')
    tasks = ['cheetah_run', 'reacher_easy', 'walker_walk']
    task2title = OrderedDict.fromkeys(tasks)
    for key in task2title:
        task2title[key] = key.replace('_', ' ').title()
    seeds = [0, 1, 2]
    single_step_qs = [True]

    titled_data = OrderedDict.fromkeys(task2title.values())
    for task in tasks:
        labeled_data = {}
        for single_step_q in single_step_qs:
            data_seeds = []
            for seed in seeds:
                if task == 'walker_walk' and seed == 1:
                    continue
                pattern = f'*{task}*seed={seed},single_step_q={single_step_q}-*/test_metrics.jsonl'
                data_seed = load_data(logdir, pattern, keys=['step', 'test/return'])
                data_seeds.append(data_seed)
            # Checks
            for data in data_seeds:
                try:
                    assert (data['step'] == data_seeds[0]['step']).all()
                    assert len(data['step']) == len(data['test/return'])
                except:
                    import ipdb; ipdb.set_trace()

            label = f'single-step Q'
            labeled_data[label] = {}
            labeled_data[label]['x'] = data_seeds[0]['step']
            labeled_data[label]['y'] = np.mean([data['test/return'] for data in data_seeds], axis=0)
            labeled_data[label]['y_std'] = np.std([data['test/return'] for data in data_seeds], axis=0)
        titled_data[task2title[task]] = labeled_data

    return titled_data

def load_tf_init():

    logdir = Path('results/tf_init/')
    tasks = ['cup_catch', 'hopper_stand', 'pendulum_swingup']
    task2title = OrderedDict.fromkeys(tasks)
    for key in task2title:
        task2title[key] = key.replace('_', ' ').title()
    seeds = [0, 1, 2]
    # single_step_qs = [True]

    titled_data = OrderedDict.fromkeys(task2title.values())
    for task in tasks:
        labeled_data = {}
        # for single_step_q in single_step_qs:
        data_seeds = []
        for seed in seeds:
            if task == 'walker_walk' and seed == 1:
                continue
            pattern = f'*{task}*seed={seed}*-*/test_metrics.jsonl'
            data_seed = load_data(logdir, pattern, keys=['step', 'test/return'])
            data_seeds.append(data_seed)
        # Checks
        for data in data_seeds:
            try:
                assert (data['step'] == data_seeds[0]['step']).all()
                assert len(data['step']) == len(data['test/return'])
            except:
                import ipdb; ipdb.set_trace()

        label = 'Our impl.-TF init'
        labeled_data[label] = {}
        labeled_data[label]['x'] = data_seeds[0]['step']
        labeled_data[label]['y'] = np.mean([data['test/return'] for data in data_seeds], axis=0)
        labeled_data[label]['y_std'] = np.std([data['test/return'] for data in data_seeds], axis=0)
        titled_data[task2title[task]] = labeled_data

    return titled_data




if __name__ == '__main__':
    setup_style()
    os.makedirs('figures/', exist_ok=True)
    # plot_results()
    # plot_update_horizon()
    # plot_single_step_q()
    # plot_dreamer_tf2()

    data = dict(
        Hello=dict(
            x=[0, 1, 2],
            y=[0, 1, 2],
            y_std=[0.1, 0.2, 0.3],
        ),
        Yes=dict(
            x=[1, 2, 3],
            y=[0, 1, 2],
            y_std=[0.3, 0.2, 0.3],
        )
    )
    titled_data = dict(Haha=data, hehe=data)
    # Figure size is in inches (2.54 cm). This is the default value.
    # fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # # ax = plot(ax=ax, data=data, title='Acrobot Swingup', xlabel=r'Frames ($\times 10^6$)', ylabel='Episode Return')
    fig, axes = plot_grid(titled_data=titled_data, ncols=2, xlabel=r'Frames ($\times 10^6$)', ylabel='Episode Return')
    plt.tight_layout()  # So that text is within the figure
    plt.savefig('./figure.png', dpi=300)
    plt.show()
