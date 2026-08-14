import unittest

import numpy as np

from protlink.negative_sampling import edge_array_to_set, sample_negative_edges
from protlink.split import split_positive_edges


def in_pair_space(edge, left_nodes, right_nodes):
    u, v = int(edge[0]), int(edge[1])
    return (u in left_nodes and v in right_nodes) or (v in left_nodes and u in right_nodes)


class NodeInductiveSplitTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 12
        self.edges = np.asarray(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (1, 11),
                (2, 8),
                (3, 9),
            ],
            dtype=np.int64,
        )
        self.split = split_positive_edges(
            self.edges,
            self.num_nodes,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=7,
            mode="node_inductive",
        )

    def test_strict_node_disjointness(self):
        train_nodes = set(self.split["train_nodes"])
        val_nodes = set(self.split["val_nodes"])
        test_nodes = set(self.split["test_nodes"])

        self.assertTrue(train_nodes.isdisjoint(val_nodes))
        self.assertTrue(train_nodes.isdisjoint(test_nodes))
        self.assertTrue(val_nodes.isdisjoint(test_nodes))

    def test_no_held_out_node_in_training_edges(self):
        train_nodes = set(self.split["train_nodes"])
        held_out = set(self.split["val_nodes"]).union(set(self.split["test_nodes"]))

        for u, v in self.split["train_pos"]:
            self.assertIn(int(u), train_nodes)
            self.assertIn(int(v), train_nodes)
            self.assertNotIn(int(u), held_out)
            self.assertNotIn(int(v), held_out)

    def test_evaluation_edges_are_unseen_to_seen(self):
        train_nodes = set(self.split["train_nodes"])
        val_nodes = set(self.split["val_nodes"])
        test_nodes = set(self.split["test_nodes"])

        for edge in self.split["val_pos"]:
            self.assertTrue(in_pair_space(edge, val_nodes, train_nodes))
        for edge in self.split["test_pos"]:
            self.assertTrue(in_pair_space(edge, test_nodes, train_nodes))

    def test_asymmetric_negative_sampling(self):
        all_pos = edge_array_to_set(self.edges)
        train_nodes = set(int(node) for node in self.split["train_nodes"])
        val_nodes = set(int(node) for node in self.split["val_nodes"])
        test_nodes = set(int(node) for node in self.split["test_nodes"])

        train_neg = sample_negative_edges(
            self.num_nodes,
            min(3, len(self.split["train_pos"])),
            all_pos,
            seed=11,
            left_candidate_nodes=self.split["train_nodes"],
            right_candidate_nodes=self.split["train_nodes"],
        )
        val_neg = sample_negative_edges(
            self.num_nodes,
            min(3, len(self.split["val_pos"])),
            all_pos,
            seed=12,
            left_candidate_nodes=self.split["val_nodes"],
            right_candidate_nodes=self.split["train_nodes"],
        )
        test_neg = sample_negative_edges(
            self.num_nodes,
            min(3, len(self.split["test_pos"])),
            all_pos,
            seed=13,
            left_candidate_nodes=self.split["test_nodes"],
            right_candidate_nodes=self.split["train_nodes"],
        )

        for edge in train_neg:
            self.assertTrue(in_pair_space(edge, train_nodes, train_nodes))
            self.assertNotIn(tuple(edge), all_pos)
        for edge in val_neg:
            self.assertTrue(in_pair_space(edge, val_nodes, train_nodes))
            self.assertNotIn(tuple(edge), all_pos)
        for edge in test_neg:
            self.assertTrue(in_pair_space(edge, test_nodes, train_nodes))
            self.assertNotIn(tuple(edge), all_pos)

    def test_backward_compatible_split_modes_return_tuple(self):
        edge_random = split_positive_edges(self.edges, self.num_nodes, seed=0, mode="edge_random")
        node_disjoint = split_positive_edges(self.edges, self.num_nodes, seed=0, mode="node_disjoint")

        self.assertIsInstance(edge_random, tuple)
        self.assertEqual(len(edge_random), 3)
        self.assertIsInstance(node_disjoint, tuple)
        self.assertEqual(len(node_disjoint), 3)


if __name__ == "__main__":
    unittest.main()
