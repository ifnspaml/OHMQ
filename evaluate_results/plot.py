import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

used_styles = []


def main():
    plt.style.use('seaborn')
    for plot_func in [
        plot_5a,
        # plot_5b,
        plot_6c,
    ]:
        global used_styles
        used_styles = []
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.set_xlabel('rate'), ax.set_ylabel('psnr')
        plot_func()
        plt.legend()
        plt.savefig(f'{plot_func.__name__}.png')
        plt.show()

fms_colors = {8: 'blue', 16: 'red', 32: 'green'}
c_markers = {0: 'o', 8: 's'}
styles = {
    'ohm': dict(ls='-', marker='o'),
    'int': dict(ls='--', marker='X'),
    'c_r=960': dict(color='grey'),
    '8 FMs': dict(color=fms_colors[8]),
    '16 FMs': dict(color=fms_colors[16]),
    '32 FMs': dict(color=fms_colors[32]),
    'cmin=0': dict(marker=c_markers[0]),
    'cmin=8': dict(marker=c_markers[8]),
    'lambda_0=0.95': dict(color='red'),
    'lambda_0=0.9': dict(color='grey'),
    'lambda_0=0.85': dict(color='black'),
}


def get_style(name=None, *, c_min=None, fms=None, lambda_0=None):
    style = {}
    names = [name, f'cmin={c_min}', f'{fms} FMs', f'lambda_0={lambda_0}']
    for n in names:
        if n in styles:
            style.update(styles[n])
            if n not in used_styles:
                used_styles.append(n)
                plt.gca().plot([], [], label=n, **({'color': 'black'} | styles[n]))
    if 'color' not in style:
        style['color'] = 'black'
    return style


def plot_5a():
    plt.suptitle('mnist adaptive')
    df = preprocess_df(pd.read_csv('mnist_ohm_adaptive_test_rd_curve.csv')).sort_values(by='ch_fraction')
    df = df[df['ch_1'] == 16]
    for ch_0, df in df.groupby('ch_0'):
        for lambda_0, df in df.groupby('lam_0'):
            for lambda_1, df in df.groupby('lam_1'):
                plt.plot(df['bitrate_real'], df['psnr'], **get_style(c_min=ch_0, lambda_0=lambda_0))


 def plot_5b():
     plt.suptitle('mnist adaptive')
     df = preprocess_df(pd.read_csv('mnist_ohm_adaptive_test_rd_curve.csv')).sort_values(by='ch_fraction')
     df = df[(df['ch_0'] == 0) & (df['lam_0'] == 0.95) & (df['lam_1'] == 0.999)]
     for ch_1, df in df.groupby('ch_1'):
         plt.plot(df['bitrate_real'], df['psnr'], **get_style('ohm', fms=ch_1))

     df = preprocess_df(pd.read_csv('mnist_int_adaptive_test_rd_curve.csv')).sort_values(by='ch_fraction')
     df = df[(df['ch_0'] == 0) & (df['lam_0'] == 0.95) & (df['lam_1'] == 0.999)]
     for ch_1, df in df.groupby('ch_1'):
         plt.plot(df['bitrate_real'], df['psnr'], **get_style('int', fms=ch_1))


def plot_6c():
    plt.suptitle('kodak adaptive')
    df = preprocess_df(pd.read_csv('kodak_ohm_adaptive_test_rd_curve.csv').sort_values(by='ch_fraction'))
    for ch_0, df in df.groupby('ch_0'):
        plt.plot(df['bitrate_real'], df['psnr'], **(get_style('ohm')) | dict(marker=None))
    df = preprocess_df(pd.read_csv('kodak_int_adaptive_test_rd_curve.csv').sort_values(by='ch_fraction'))
    for ch_0, df in df.groupby('ch_0'):
        plt.plot(df['bitrate_real'], df['psnr'], **(get_style('int')) | dict(marker=None))


def preprocess_df(df):
    for name, abbr in [('bottleneck_channels', 'ch'), ('lambda_rd', 'lam')]:
        if name in df.columns and isinstance(df[name][0], str):
            for i in [0, 1]:
                df[f'{abbr}_{i}'] = df[name].apply(lambda x: eval(x)[i])
    return df


if __name__ == '__main__':
    main()

