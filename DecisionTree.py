import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any, Dict, NamedTuple, Union, Optional, TypeVar
from collections import Counter, defaultdict

root = os.path.join('', '../machine failure')
data = os.path.join(root, 'train_set.csv')
T = TypeVar('T')
df = pd.read_csv(data, index_col=[0]).drop('my_Class', axis=1)

inputs = list(df.itertuples(name='Row', index=False))

def entropy(class_prob: List[T]) -> float:
    return sum(-p*math.log2(p) for p in class_prob)

def class_prob(labels: List[T]) -> List[float]:
    total = len(labels)
    return [count/total for count in Counter(labels).values()]

def data_entropy(labels: List[T]) -> float:
    return entropy(class_prob(labels))

def split_entropy(subsets: List[List[T]]) -> List[float]:
    total = sum(len(subset) for subset in subsets)
    return sum([data_entropy(subset)*len(subsets)/total for subset in subsets])

def split_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)
        partitions[key].append(input)
    return partitions

def split_entropy_by(inputs: List[T], attribute: str, label_attr: str) -> float:
    splits = split_by(inputs, attribute)
    return [[[input, label_attr] for input in split] for split in splits.values()]

f = split_entropy_by(inputs, 'Outlook', 'my_class')

class Leaf(NamedTuple):
    value: T
class Split(NamedTuple):
    attribute: str
    subtrees: dict
    defaultValue: T = None

DecisionTree = Union[Leaf, Split]
def build_id3Tree(inputs: List[T], split_attributes: List[str], target_attribute: str) -> DecisionTree:
    labelCounts = Counter(getattr(input, target_attribute) for input in inputs)
    mostCommon = labelCounts.most_common(1)[0][0]
    if len(labelCounts) == 1:
        return Leaf(mostCommon)
    if not split_attributes:
        return Leaf(mostCommon)
    
    #helper function to find best attribute to split on
    def splitEntropy(attribute: str) -> float:
        return split_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)
    partitions = split_by(inputs, best_attribute)
    new_attributes = [atr for atr in split_attributes if atr!=best_attribute]
    # apply recursively to build the subtrees
    subtrees = {atrValue : build_id3Tree(subset, new_attributes, target_attribute)
                                        for (atrValue, subset) in partitions.items()}
    return Split(best_attribute, subtrees, defaultValue=mostCommon)
built = build_id3Tree(inputs, ['Outlook', 'Temp', 'Humidity', 'Windy'], 'my_class')

def classify(tree: DecisionTree, input:Any) -> Any:
    if isinstance(tree, Leaf):
        return tree.value
    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees:
        return tree.defaultValue
    subtree = tree.subtrees[subtree_key]
    return classify(subtree, input)




class Row(NamedTuple):
    Outlook: str
    Temp: str
    Humidity: bool
    Windy: bool
print(inputs[0])
e = Row('sunny', 'hot', True, True)

print(classify(built, e))