import systems_fun as sf
import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering


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

class TestFindEquilibria:
    def rhs(self,X,params):
        x,y=X
        a,b,c1,c2=params
        return [x*(y-a*x), y*(x+ b+ y)]

    def rhsJac(self,X, params):
        x, y = X
        a, b, c1, c2 = params
        return np.array([[y - 2 * a * x, x], [y, b + x + 2 * y]])



    def analyticFind(self,params):
        a,b,c1,c2=params
        #nSinks, nSaddles, nSources, nNonRough
        if ((a>0 and b>0 )  or (a<-1 and b>0 )):
            result = (1,1,0,1)
        elif((a>0 and b<0 )or(a<-1 and b<0) ):
            result = (0,1,1,1)
        elif(-1<a<0 and b>0):
            result = (1, 0, 0, 2)
        elif (-1 < a < 0 and b > 0):
            result = (0, 0, 1, 2)
        elif(a == -1 and b>0):
            result= (0)
        elif (a == -1 and b < 0):
            result = (0)
        elif (a == -1 and b == 0):
            result = (0,0,0,1)
        elif (a < -1 and b == 0 or-1< a < 0 and b == 0 or a > 0 and b == 0):
            result = (0,0,0,1)
        elif (-1< a < 0 and b == 0):
            result = (0,0,0,1)
        return result

    bounds = [(-3.5, 3.5), (-3.5, 3.5)]
    borders = [(-3.5 , 3.5 ), (-3.5 , 3.5)]

    def test_FindEqInSinglePoint(self):
        ud = [-0.5,0,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders,'ShgoEqFinder',(1000, 100,1e-15))
        data= res[:,2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)


    def test_FindEqInSinglePoint2(self):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders, 'ShgoEqFinder',(1000, 100,1e-15))
        data= res[:,2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePoint3(self):
        ud = [-1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders,'ShgoEqFinder',(300, 30,4e-14))
        X = res[:, 0:2]
        if list(X) and len(list(X)) != 1:
            clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                                 distance_threshold=(1e-5))
            clustering.fit(X)
            data = res[:, 2:5]
            trueStr = sf.mergePoints(clustering.labels_, data)
            data = res[list(trueStr), 2:5]
        else:
            data = res[:, 2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEq2InSinglePointNewton(self):
        ud = [-0.5,0,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders, 'NewtonEqFinder',(21, 21,1e-15))
        X = res[:, 0:2]
        if list(X)and len(list(X))!= 1:
            clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                                 distance_threshold=(1e-5))
            clustering.fit(X)
            data = res[:, 2:5]
            trueStr = sf.mergePoints(clustering.labels_, data)
            data= res[list(trueStr),2:5]
        else:
            data = res[:, 2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewton2(self):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders, 'NewtonEqFinder',(71, 71,1e-18))
        X = res[:, 0:2]
        if list(X)and len(list(X))!= 1:
            clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                                 distance_threshold=(1e-5))
            clustering.fit(X)
            data = res[:, 2:5]
            trueStr = sf.mergePoints(clustering.labels_, data)
            data= res[list(trueStr),2:5]
        else:
            data = res[:, 2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewton3(self):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders, 'NewtonEqFinder',(71, 71,1e-18))
        X = res[:, 0:2]
        if list(X)and len(list(X))!= 1:
            clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                                 distance_threshold=(1e-5))
            clustering.fit(X)
            data = res[:, 2:5]
            trueStr = sf.mergePoints(clustering.labels_, data)
            data= res[list(trueStr),2:5]
        else:
            data = res[:, 2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewtonUp(self):
        ud = [-0.5,0,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders, 'NewtonEqFinderUp',(81, 81,1e-20))
        data = res[:, 2:5]
        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewtonUp2(self):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders, 'NewtonEqFinderUp',(71, 71,1e-16))
        X = res[:, 0:2]
        if list(X) and len(list(X)) != 1:
            clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                                 distance_threshold=(1e-5))
            clustering.fit(X)
            data = res[:, 2:5]
            trueStr = sf.mergePoints(clustering.labels_, data)
            data = res[list(trueStr), 2:5]
            print(res[list(trueStr)])
        else:
            data = res[:, 2:5]

        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewtonUp3(self):
        ud = [-1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds,  self.borders,  'NewtonEqFinderUp',(101, 101,8e-17))
        X = res[:, 0:2]
        if list(X) and len(list(X)) != 1:
            clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                                 distance_threshold=(1e-5))
            clustering.fit(X)
            data = res[:, 2:5]
            trueStr = sf.mergePoints(clustering.labels_, data)
            data = res[list(trueStr), 2:5]
            print(res[list(trueStr)])
        else:
            data = res[:, 2:5]

        describe = sf.describePortrType(data.tolist())
        assert describe == self.analyticFind(ud)

class TestCreateDistMatrix:

    def testMatrix1(self):
        coords = [[1, 1], [1, 1]]
        a = np.array(coords)
        assert sf.createDistMatrix(a).tolist() == [[0,0],[0,0]]

    def testMatrix2(self):
        coords = [[0, 0], [2, 2]]
        a = np.array(coords)
        assert sf.createDistMatrix(a).tolist() == [[0,2*np.sqrt(2)],[2*np.sqrt(2),0]]

    def testMatrix3(self):
        coords = [[0, 1], [0, 0]]
        a = np.array(coords)
        assert sf.createDistMatrix(a).tolist() == [[0,1],[1,0]]

    def testMatrix4(self):
        coords = [[2, 2], [2, 2]]
        a = np.array(coords)
        assert sf.createDistMatrix(a).tolist() == [[0, 0], [0, 0]]

    def testWORK(self):
        t = 2
        matrix = [[0,1.1,1.1,2],[1.1,0,0,1],[1.1,0,0,1],[2,1.1,1.1,0]]
        assert  sf.work(matrix,1.0) == [0,1,1,1]
