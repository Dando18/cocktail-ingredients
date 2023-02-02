''' Find an ideal set of ingredients to cover a large number of recipes.
    author: Daniel Nichols
    date: December 2022

    notes:
    
        There are a couple of ways to phrase the problem here. Let I_u and R_u
        be the universal sets of ingredients and recipes, respectively.
        (1) Given k, we want to select a subset I* of I_u of size k that 
            satisfies the most possible recipes in R_u.
        (2) Given k, select k recipes that require the least number of
            ingredients.
        (3) How many ingredients do we need to satisfy all of R_u?
        (4) 2 & 3 but given a fixed set of ingredients or recipes.
'''
# std imports
from argparse import ArgumentParser
from collections import Counter
import copy
from dataclasses import dataclass
from functools import partial
import hashlib
from itertools import combinations
import json
import logging
from math import comb, log
import multiprocessing as mp
import os
from random import sample
import time
from typing import Iterable, Tuple

# tpl imports
from alive_progress import alive_it
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


@dataclass
class Ingredient:
    name: str
    amount: str
    is_garnish: bool = False

    def __eq__(self, other):
        if not isinstance(other, Ingredient):
            raise TypeError
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class Recipe:
    name: str
    ingredients: frozenset

    def uses_ingredient(self, ingr: Ingredient, name_only: bool = True) -> bool:
        if name_only:
            return any(i.name == ingr.name for i in self.ingredients)
        else:
            return ingr in ingredients

    def can_be_made_with(self, ingrs: Iterable[Ingredient]) -> bool:
        return all(i in ingrs for i in self.ingredients)

    def get_num_shared_ingredients(self, recipe) -> int:
        return len(frozenset.intersection(self.ingredients, recipe.ingredients))

    def __eq__(self, other):
        if not isinstance(other, Recipe):
            raise TypeError
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


def remove_items_in_pantry(recipes: Iterable[Recipe], pantry: Iterable[str]) -> Iterable[Recipe]:
    recipes = copy.deepcopy(recipes)
    for recipe in recipes:
        recipe.ingredients = frozenset(filter(lambda i: i.name not in pantry, recipe.ingredients))
    return recipes


def remove_garnishes(recipes: Iterable[Recipe]) -> Iterable[Recipe]:
    recipes = copy.deepcopy(recipes)
    for recipe in recipes:
        recipe.ingredients = frozenset(filter(lambda i: not i.is_garnish, recipe.ingredients))
    return recipes


def make_substitutions(recipes: Iterable[Recipe], subs: dict) -> Iterable[Recipe]:
    def _get_sub(ingr: Ingredient) -> Ingredient:
        for key, values in subs.items():
            if ingr.name == key or ingr.name in values:
                ingr.name = key
                return ingr 
        return ingr

    recipes = copy.deepcopy(recipes)
    for recipe in recipes:
        recipe.ingredients = frozenset(map(_get_sub, recipe.ingredients))
    return recipes


def _set_intersection_size(sets: Iterable[set]):
    ''' Return the intersection size of sets.
    '''
    return len(frozenset.intersection(*sets))


def _maximum_k_intersections(subsets: Iterable[set], k: int) -> Tuple[set]:
    ''' Find the largest possible intersection of k sets.
    '''
    vals = alive_it(combinations(subsets, k), title='Finding max intersection', total=comb(len(subsets), k))
    return max(vals, key=_set_intersection_size)


def partition(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _maximum_k_intersections_mp(subsets: Iterable[set], k: int, np: int = 4) -> Tuple[set]:
    ''' Find the largest possible intersection of k sets. Use multiprocessing.
    '''
    inter_max = partial(max, key=_set_intersection_size)
    with mp.Pool(np) as pool:
        partial_maxes = pool.map(inter_max, partition(list(combinations(subsets, k)), np))
    return max(partial_maxes, key=_set_intersection_size)


def minimum_recipe_union(recipes: Iterable[Recipe], k: int):
    ''' Find the smallest set of ingredients that satisfy at least k recipes.
        Greedy approximation.
    '''
    recipes = copy.deepcopy(recipes)
    union = set()
    for _ in alive_it(range(k)):
        s = min(recipes, key=lambda x: len(union | x.ingredients))
        recipes.remove(s)
        union |= s.ingredients
    return union


def get_recipe_degree(recipe: Recipe, recipes: Iterable[Recipe], scale: bool = False) -> int:
    ''' Get the number of recipes that share an ingredient with `recipe`. If scale, then
        weight each edge by the number of shared ingredients.

        Args:
            recipe: the recipe to compute the degree for
            recipes: all other recipes in graph
            scale: if True, then use size of intersection in degree sum

        Return:
            The total number of recipes with shared ingredient(s).
    '''
    degree = 0
    for r in recipes:
        inc = recipe.get_num_shared_ingredients(r)
        if not scale and inc != 0:
            degree += 1
        else:
            degree += inc
    return degree


def get_ingredient_degree(ingr: Ingredient, recipes: Iterable[Recipe], scale: bool = True) -> int:
    ''' The number of ingredients that `ingr` shares a recipe with. In a graph G=(V,E) where the vertices V
        are the ingredients and an edge in E connects two vertices if their corresponding ingredients
        share a recipe. If scale, then each edge is weighted by the number of recipes shared.

        Args:
            ingr: ingredient to calculate degree of
            recipes: list of all recipes
            scale: if True, then use number of shared recipes to scale degree

        Return:
            the degree of ingr on G
    '''
    edges = Counter(i for r in recipes for i in r.ingredients if ingr in r.ingredients and i != ingr)
    return sum(edges.values()) if scale else len(edges)


def get_obj_md5(obj):
    return hashlib.md5(json.dumps(obj).encode('utf-8')).hexdigest()


def find_best_ingredients_subset(
    recipes: Iterable[Recipe],
    k: int,
    method: str = 'exhaustive'
) -> Tuple[Iterable[Ingredient], Iterable[Recipe]]:
    ''' Given k, find a set of k ingredients that satisfies the most possible recipes.
        https://math.stackexchange.com/a/2786746/274085

        Args:
            recipes: a list of Recipe objects to search through.
            k: the total number of ingredients to use.
            method: one of exhaustive or heuristic
        
        Return:
            The list of ingredients and a list of recipes that they satisfy.
    '''
    # first get a unique list of ingredients from all recipes
    recipes = copy.deepcopy(recipes)
    all_ingredients = set()
    for recipe in recipes:
        all_ingredients.update(recipe.ingredients)
    
    # make sure data is good
    min_k = min(len(r.ingredients) for r in recipes)
    method = method.lower()
    assert k >= 0, f'k must be a positive integer: {k} given'
    assert k < len(all_ingredients), f'k is larger than the total number of ingredients: {k} > {len(all_ingredients)}'
    assert k >= min_k, f'k={k} but the smallest recipe has {min_k} ingredients'
    assert method in ['exhaustive', 'greedy', 'greedy2', 'greedy3', 'greedy4', 'random', 'heuristic', 'cluster'], \
        f'Unknown solving method: {method}'

    if method == 'exhaustive':
        # create list of subsets S_i for each ingredient i; S_i is all recipes that can be made without i 
        subsets = list(frozenset(r for r in recipes if not r.uses_ingredient(i)) for i in all_ingredients)

        # find maximum intersection of subsets
        max_subsets = _maximum_k_intersections(subsets, len(subsets) - k)

        final_recipes = list(frozenset.intersection(*max_subsets))
        ingredients_to_use = list(frozenset.union(*(r.ingredients for r in final_recipes)))
        return ingredients_to_use, final_recipes
    elif method == 'greedy':
        ingr_counts = Counter(i for r in recipes for i in r.ingredients)
        final_ingredients = [i for i,_ in ingr_counts.most_common(k)]
        final_recipes = [r for r in recipes if r.can_be_made_with(final_ingredients)]
        return final_ingredients, final_recipes
    elif method == 'greedy2':
        recipe_scores = [(r, get_recipe_degree(r, recipes, scale=True)) for r in alive_it(recipes)]
        recipe_scores.sort(key=lambda x: x[1])
        final_ingredients = set()
        while len(final_ingredients) <= k:
            back = recipe_scores.pop()
            final_ingredients.update(back[0].ingredients)
        final_recipes = [r for r in recipes if r.can_be_made_with(final_ingredients)]
        return final_ingredients, final_recipes
    elif method == 'greedy3':
        ingr_scores = [(i, get_ingredient_degree(i, recipes, scale=True)) for i in alive_it(all_ingredients)]
        ingr_scores.sort(key=lambda x: x[1], reverse=True)
        final_ingredients = [x[0] for x in ingr_scores[:k]]
        final_recipes = [r for r in recipes if r.can_be_made_with(final_ingredients)]
        return final_ingredients, final_recipes
    elif method == 'greedy4':
        final_ingredients = minimum_recipe_union(recipes, k)
        final_recipes = [r for r in recipes if r.can_be_made_with(final_ingredients)]
        return final_ingredients, final_recipes
    elif method == 'random':
        NITER = 10000
        samples = [sample(all_ingredients, k) for _ in alive_it(range(NITER), title='Sampling')]
        best = max(alive_it(samples, title='Testing'), key=lambda s: sum([1 for r in recipes if r.can_be_made_with(s)]))
        final_recipes = [r for r in recipes if r.can_be_made_with(best)]
        return best, final_recipes
    elif method == 'cluster':
        cluster = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        X = [r.ingredients for r in recipes]
        distance_matrix = np.array([[float(len(i & j)) for j in X] for i in X])
        distance_matrix = np.max(distance_matrix) - distance_matrix
        
        cluster.fit(distance_matrix)
        union = set()
        for c in range(k):
            union |= recipes[cluster.labels_.tolist().index(c)].ingredients
        final_recipes = [r for r in recipes if r.can_be_made_with(union)]
        return union, final_recipes
    elif method == 'heuristic':
        raise NotImplementedError('heuristic method not yet implemented')

    return [], []


def main():
    parser = ArgumentParser(description='recipe/ingredients optimizations')
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO', type=str.upper, help='logging level')
    parser.add_argument('-o', '--output', type=str, help='path to csv to write or append results')
    parser.add_argument('-r', '--recipes', type=str, required=True, help='path to recipes file')
    parser.add_argument('-p', '--pantry', type=str, help='path to pantry file')
    parser.add_argument('--ignore-garnish', action='store_true', help='ignore garnishes in recipes')
    parser.add_argument('--allow-substitutes', action='store_true', help='allow substitutions')
    parser.add_argument('-k', type=int, default=3, help='meta-parameter for different algorithms')
    parser.add_argument('--find-ingredients', type=str.lower, nargs='?', const='exhaustive', 
        choices=['exhaustive', 'heuristic', 'greedy', 'greedy2', 'greedy3', 'greedy4', 'random', 'cluster'], 
        help='given k, find max number of recipes made with k ingredients.')
    args = parser.parse_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', level=numeric_level)

    # read input
    with open(args.recipes, 'r') as fp:
        recipes_json = json.load(fp)

    recipes = []
    for recipe in recipes_json['recipes']:
        ingredients = []
        for i in recipe['ingredients']:
            is_garnish = ('garnish' in i) and (i['garnish'].lower() == 'true')
            ingredients.append(Ingredient(name=i['name'], amount=i['amt'], is_garnish=is_garnish))
        
        recipes.append(Recipe(name=recipe['name'], ingredients=set(ingredients)))

    if args.pantry:
        with open(args.pantry, 'r') as fp:
            pantry = json.load(fp)
        
        recipes = remove_items_in_pantry(recipes, pantry['ingredients'])
    
    if args.ignore_garnish:
        recipes = remove_garnishes(recipes)
    
    if args.allow_substitutes:
        recipes = make_substitutions(recipes, recipes_json['substitutes'])

    if args.find_ingredients:
        start = time.time()
        ingr, rec = find_best_ingredients_subset(recipes, args.k, method=args.find_ingredients)
        duration = time.time() - start
        
        print('With {} ingredient(s):\n\t{}'.format(len(ingr), ', '.join(i.name for i in ingr)))
        print('You can make {} drink(s):\n\t{}'.format(len(rec), ', '.join(r.name for r in rec)))
        logging.debug(f'Found ingredients in {duration} second(s).')

        if args.output:
            out = {'method': [args.find_ingredients], 'k': [args.k], 'num_ingredients': [len(ingr)], 
                'num_recipes': [len(rec)], 'duration': [duration], 'dataset_md5': [get_obj_md5(recipes_json)],
                'pantry_md5': [get_obj_md5(pantry) if pantry else ''], 'filter_pantry': [bool(args.pantry)], 
                'filter_garnish': [args.ignore_garnish], 'make_substitutes': [args.allow_substitutes]}
            pd.DataFrame(out).to_csv(args.output, index=False, mode='a', header=not os.path.exists(args.output))


if __name__ == '__main__':
    main()