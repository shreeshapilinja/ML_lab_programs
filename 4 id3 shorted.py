# some what shortened
import pandas as pd
import math

def load_csv(filename):
    df = pd.read_csv(filename)
    dataset = df.values.tolist()
    headers = df.columns.tolist()
    return dataset, headers

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        self.answer = ""

def entropy(S):
    counts = S.value_counts(normalize=True)
    entropy = -(counts * counts.apply(math.log2)).sum()
    return entropy

def compute_gain(data, col, target_col):
    total_entropy = entropy(data[target_col])
    attr_counts = data[col].value_counts()

    entropies = data.groupby(col).apply(lambda x: entropy(x[target_col]))
    ratios = attr_counts / len(data)

    weighted_entropy = (ratios * entropies).sum()
    gain = total_entropy - weighted_entropy

    return gain

def subtables(data, col):
    subtables = {val: data[data[col] == val].reset_index(drop=True) for val in data[col].unique()}
    return list(subtables.keys()), subtables

def build_tree(data, features, target_col):
    unique_labels = data[target_col].unique()

    if len(unique_labels) == 1:
        node = Node("")
        node.answer = unique_labels[0]
        return node

    gains = [compute_gain(data, col, target_col) for col in features]
    split = gains.index(max(gains))
    attribute = features[split]

    node = Node(attribute)
    attr_values, subtables_dict = subtables(data, attribute)

    for value, subtable in subtables_dict.items():
        if len(subtable) == 0:
            child = Node("")
            child.answer = data[target_col].mode()[0]
        else:
            child = build_tree(subtable.drop(attribute, axis=1), features[:split] + features[split+1:], target_col)
        node.children[value] = child

    return node

def print_tree(node, level=0):
    if node.answer:
        print(" " * level + node.answer)
        return
    print(" " * level + node.attribute)
    for value, child in node.children.items():
        print(" " * (level + 1) + str(value))
        print_tree(child, level + 2)

def classify(node, x_test, features):
    if node.answer:
        print(node.answer)
        return
    attribute = node.attribute
    pos = features.index(attribute)
    value = x_test[pos]
    if value in node.children:
        classify(node.children[value], x_test, features)

dataset, features = load_csv("id3.csv")
df = pd.DataFrame(dataset, columns=features)
target_col = features[-1]  # Assuming the target column is the last column
features = features[:-1]  # Excluding the target column from features
node1 = build_tree(df, features, target_col)
print("The decision tree for the dataset using ID3 algorithm is:")
print_tree(node1)

testdata, _ = load_csv("id3_test_1.csv")
df_test = pd.DataFrame(testdata, columns=features)
for _, xtest in df_test.iterrows():
    print("The test instance:", xtest.values)
    print("The label for the test instance:", end=" ")
    classify(node1, xtest, features)
