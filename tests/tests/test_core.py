import numpy as np

from declip.metrics import multilabel_metrics
from declip.model import DeCLIP
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

