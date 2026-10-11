import numpy as np
import torch
from torch.utils.data import Subset

from declip.metrics import multilabel_metrics
from declip.model import DeCLIP
from declip.old_labels import diagnose_old_label_cache, select_old_labels
from declip.protocols import voc_protocol


def test_protocols():
    b0 = voc_protocol("b0-c4")
    b4 = voc_protocol("b4-c2")
    assert [len(x) for x in b0.tasks] == [4, 4, 4, 4, 4]
    assert [len(x) for x in b4.tasks] == [4, 2, 2, 2, 2, 2, 2, 2, 2]
    assert len(b4.seen(8)) == 20


def test_ast():
    assert DeCLIP.ast_temperature(1, 4) == 1
    assert DeCLIP.ast_temperature(2, 6) == 16 * 2 / 6
    assert DeCLIP.ast_temperature(2, 50) == 1


def test_metrics_perfect():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    p = np.array([[.9, .1], [.1, .9], [.8, .8]])
    result = multilabel_metrics(y, p)
    assert result["mAP"] == 100
    assert result["CF1"] == 100
    assert result["OF1"] == 100


def test_pseudo_labels_need_both_sides_and_respect_budgets():
    probabilities = torch.tensor([
        [.99, .99], [.995, .995], [.98, .98],
        [.01, .01], [.005, .005], [.015, .015], [.02, .02],
        [.5, .5], [.99, .1], [.5, .5],
    ]).unsqueeze(1)
    targets, weights, summary = select_old_labels(
        probabilities, positive_fraction=.2, negative_fraction=.3)
    assert summary["positive_count"] == [2]
    assert summary["negative_count"] == [3]
    assert (weights[:, 0] > 0).sum().item() == 5
    assert weights[8, 0] == 0  # A confident prediction in one view is insufficient.
    positive_mass = weights[(targets == 0) & (weights > 0)].sum()
    negative_mass = weights[(targets == 1) & (weights > 0)].sum()
    assert negative_mass <= positive_mass + 1e-6


def test_pseudo_labels_do_not_train_on_one_sided_confidence():
    probabilities = torch.full((20, 2, 2), .999)
    targets, weights, summary = select_old_labels(probabilities)
    assert weights.sum() == 0
    assert summary["positive_count"] == [0, 0]
    assert summary["negative_count"] == [0, 0]


def test_diagnostics_read_labels_without_loading_images():
    class LabelOnlyDataset:
        ids = ["first", "second"]
        labels = torch.tensor([[1.0], [0.0]])

        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("diagnostics must not load image pixels")

    cache = {"second": (torch.tensor([1]), torch.tensor([.1]))}
    report = diagnose_old_label_cache(cache, Subset(LabelOnlyDataset(), [1]), [0])
    assert report["negative_precision"] == 1.0
    assert report["negative_selected"] == 1

