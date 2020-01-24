import random, unittest
from DT_RF_module.solution_AG import DecisionTree,RandomForest

class Test(unittest.TestCase):
  """
  We are using Python's unittest framework here, any method on this class prepended
  with `test_` will run.
  """

  def generate_data_and_labels(self):

    n_dimensions = random.randint(2,10)
    n_data = random.randint(5,100)

    data = []
    labels = []
    for i in range(n_data):
      labels.append(random.randint(0,2))
      feature_vector = []
      for j in range(n_dimensions):
        feature_vector.append(random.uniform(-100,100))
      data.append(feature_vector)

    return data, labels

  def test_trivial_data(self):
    """
    All labels the same
    """

    tree = DecisionTree()
    data = [[1,2,3], [4,5,6], [7,8,9]]
    labels = [1,1,1]
    tree.fit(data, labels)

    assert tree.predict([2,8,9]) == 1

  def test_trivial_data_forest(self):
    """
    All labels the same
    """

    forest = RandomForest(n_trees=7)
    data = [[1,2,3], [4,5,6], [7,8,9], [10,8,9]]
    labels = [1,1,1,1]
    forest.fit(data, labels)

    assert forest.predict([2,8,9]) == 1

  def test_training_data_correctly_self_classified(self):
    """
    Tests that after a tree is built from training data, running that same training data is assigned
    correct labels by the predict method
    """

    test_data = [
      [-1, 0, 1],
      [1, 2, 3],
      [11, 3, 21],
      [0,-9, 27]
    ]

    test_labels = [0, 0, 1, 1]

    tree = DecisionTree()
    tree.fit(test_data, test_labels)

    for sample, label in zip(test_data, test_labels):
      assert tree.predict(sample) == label

  def test_very_simple_linearly_separable_data(self):
    """
    Trains a tree on simple data where everything is separated by the x=0 hyperplane
    (the labels correspond to whether the first dimension of the feature vector is negative or positive)
    """

    train_data = [
        [-10, 5, -4, -9],
        [-1, -8, 3, 12],
        [-2, 0, 1, 2],
        [6, 2, -2, 2],
        [12, 0, 0, 0]
    ]

    train_labels = [0, 0, 0, 1, 1]

    tree = DecisionTree()
    tree.fit(train_data, train_labels)

    test_data = [
        [-20, 1, 2, 3],
        [110, 0, -1, -10],
        [7, 1, 2, 3],
        [-56, 1, 0.01, 2]
    ]

    test_labels = [0, 1, 1, 0]

    for sample, label in zip(test_data, test_labels):
        assert tree.predict(sample) == label

  def test_all_training_data_correctly_classified(self):
    """
    Generates data of random dimension and length
    """
    tree = DecisionTree()
    data, labels = self.generate_data_and_labels()

    tree.fit(data, labels)

    for sample, label in zip(data, labels):

      assert tree.predict(sample) == label

  def test_very_simple_linearly_separable_data_forest(self):
    """
    Trains a forest on simple data, making sure the output conforms to
    specified API. No check on the output since there is randomness involved.
    """
    train_data = [
        [-10, 5, -4, -9],
        [-1, -8, 3, 12],
        [-2, 0, 1, 2],
        [6, 2, -2, 2],
        [12, 0, 0, 0]
    ]

    train_labels = [0, 0, 0, 1, 1]

    forest = RandomForest(n_trees=3)
    forest.fit(train_data, train_labels)
    assert forest.predict([1, 2, 3, 4]) in (0, 1)


if __name__ == '__main__':
    unittest.main()
