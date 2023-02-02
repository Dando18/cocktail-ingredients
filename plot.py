''' Plot results for tests.
    author: Daniel Nichols
    date: January 2023
'''
# std imports
from argparse import ArgumentParser

# tpl imports
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


def plot_scores(df: pd.DataFrame):
    ''' Plot method scores against each other
    '''
    plt.clf()
    sns.lineplot(data=df, x='num_ingredients', y='num_recipes', hue='method')
    plt.xlabel('# Ingredients Allowed')
    plt.ylabel('# Recipes Covered')
    plt.title('Comparing Methods for Finding Best Ingredient Subset')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.show()


def plot_duration(df: pd.DataFrame):
    ''' Plot method duration against each other
    '''
    plt.clf()
    sns.lineplot(data=df, x='num_ingredients', y='duration', hue='method')
    plt.xlabel('# Ingredients Allowed')
    plt.ylabel('Runtime (s)')
    plt.title('Comparing Runtime of Methods')
    plt.yscale('log', base=10)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.show()


def main():
    parser = ArgumentParser(description='plot results plots')
    parser.add_argument('-i', '--input', type=str, required=True, help='input dataset')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    sns.set()
    plot_scores(df)
    plot_duration(df)



if __name__ == '__main__':
    main()