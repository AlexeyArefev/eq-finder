import systems_fun as sf
import cmath as cm
import numpy as np
import pytest


class TestDescribeEqType:
    def test_saddle(self):
        assert sf.describeEqType(np.array([-1, 1])) == (1, 0, 1, 0, 0)

    def test_stable_node(self):
        assert sf.describeEqType(np.array([-1, -1])) == (2, 0, 0, 0, 0)

    def test_stable_focus(self):
        assert sf.describeEqType(np.array([-1 + 1j, -1 - 1j])) == (2, 0, 0, 1, 0)

    def test_unstable_node(self):
        assert sf.describeEqType(np.array([+1, +1])) == (0, 0, 2, 0, 0)

    def test_unstable_focus(self):
        assert sf.describeEqType(np.array([1 + 1j, 1 - 1j])) == (0, 0, 2, 0, 1)

    def test_passingTuple(self):
        # rewrite as expecting some exception
        with pytest.raises(TypeError):
            sf.describeEqType([1 + 1j, 1 - 1j]) == (0, 0, 2, 0, 1)
        # assert sf.describeEqType([1+1j, 1-1j])==(0, 0, 2, 0, 1)

    def test_almost_focus(self):
        assert sf.describeEqType(np.array([-1e-15 + 1j, -1e-15 - 1j])) == (0, 2, 0, 0, 0)

    def test_center(self):
        assert sf.describeEqType(np.array([1j, - 1j])) == (0, 2, 0, 0, 0)

class TestISComplex:
    def test_complex(self):
        assert sf.isComplex(1+2j) == 1

    def test_real(self):
        assert sf.isComplex(1) == 0

    def test_complex_without_real(self):
        assert sf.isComplex(8j) == 1

class TestInBounds:
    def test_inBounds(self):
        assert sf.inBounds((1,1),[(0,2),(0,2)]) == 1

    def test_onBounds(self):
        assert sf.inBounds((1,1),[(1,2),(0,2)]) == 0

    def test_outBounds(self):
        assert sf.inBounds((5,5),[(1,1),(0,6)]) == 0

    def test_outBoundForAction(self):
        assert sf.inBounds((5, 6), [(1, 1), (0, 6)]) == 0