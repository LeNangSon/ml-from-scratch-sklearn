import numpy as np 


def _allocate_samples(class_counts, n_samples, rng):
    total = class_counts.sum()
    expected = class_counts / total * n_samples
    allocated = np.floor(expected).astype(np.int32)
    remaining = n_samples - allocated.sum()
    remainders = expected - allocated
    for _ in range(remaining):
        max_remainder = remainders.max()

        candidates = np.flatnonzero(
            np.isclose(remainders, max_remainder)
        )

        selected = rng.choice(candidates)

        allocated[selected] += 1
        remainders[selected] = -1

    return allocated

def train_test_split(X, y, test_size = 0.2, random_state = None):

    rng = np.random.default_rng(random_state)
    
    classes = np.unique(y)

    indexed_class  = []
    for i in classes:
        class_i = np.where(y == i)      # np.where -> tuple 
        indexed_class.append(class_i[0])   

    class_counts = np.array([i.size for i in indexed_class])

    shuffled_class  = []
    for i in indexed_class:
        shuffled_class.append(rng.permutation(i))

    n_samples = np.ceil(y.size * test_size)
    allocated = _allocate_samples(class_counts, n_samples, rng)

    train_index = []
    test_index = []
    for i, j in zip(shuffled_class, allocated):
        train_index.append(i[j:])
        test_index.append(i[:j])

    train_index = np.concatenate(train_index)
    test_index = np.concatenate(test_index)

    # shuffle
    train_index = rng.permutation(train_index)
    test_index = rng.permutation(test_index)
    return [X[train_index], X[test_index], y[train_index], y[test_index]]