__author__ = 'c11tch'
import unittest
import numpy as np
from GraphMng import GraphMng
from localTest import localTest
from scipy.sparse import lil_matrix


class GraphMngTestCase(unittest.TestCase):
    GraphMng = GraphMng()
    localTest = localTest()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_MakeUserMat(self):
        for i in range(2):
            self.GraphMng.AddNewUser(i)
        self.GraphMng.makeUserMat()
        self.assertEqual(self.GraphMng.matrix_UU.shape, (2, 2))

    def test_MakeTagMat(self):
        for i in range(4):
            self.GraphMng.AddNewTag(i)
        self.GraphMng.makeTagMat()
        self.assertEqual(self.GraphMng.matrix_TT.shape, (4, 4))

    def test_MakeItemMat(self):
        for i in range(5):
            self.GraphMng.AddNewItem(i)
        self.GraphMng.makeItemMat()
        self.assertEqual(self.GraphMng.matrix_II.shape, (5, 5))

    def test_UU(self):
        self.assertEqual(self.localTest.GraphMng.matrix_UU.shape, (2, 2))

    def test_II(self):
        self.assertEqual(self.localTest.GraphMng.matrix_II.shape, (3, 3))

    def test_TT(self):
        self.assertEqual(self.localTest.GraphMng.matrix_TT.shape, (4, 4))

    def test_UI(self):
        expected = [[0.5, 0.5, 0],
                    [0, 0, 0.5]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.matrix_UI
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_IU(self):
        expected = [[0.5, 0],
                    [0.5, 0],
                    [0, 0.5]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.matrix_IU
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_UT(self):
        expected = [[0.5, 0.5, 0.5, 0],
                    [0.5, 0.5, 0, 0.5]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.matrix_UT
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_TU(self):
        expected = [[0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0],
                    [0, 0.5]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.matrix_TU
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_IT(self):
        expected = [[0.5, 0.5, 0, 0],
                    [0, 0, 0.5, 0],
                    [0.5, 0.5, 0, 0.5]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.matrix_IT
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_TI(self):
        expected = [[0.5, 0, 0.5],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0],
                    [0, 0, 0.5]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.matrix_TI
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def makeTestMatrix(self):
        a = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24, 25, 26, 27],
            [28, 29, 30, 31, 32, 33, 34, 35, 36],
            [37, 38, 39, 40, 41, 42, 43, 44, 45],
            [46, 47, 48, 49, 50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59, 60, 61, 62, 63],
            [64, 65, 66, 67, 68, 69, 70, 71, 72],
            [73, 74, 75, 76, 77, 78, 79, 80, 81]]
        a = lil_matrix(a).tocsr()
        return a

    def test_ColumnNomalization(self):
        b = [[0.003003003, 0.005847953, 0.008547009, 0.011111111, 0.013550136, 0.015873016, 0.018087855, 0.02020202, 0.022222222],
            [0.03003003, 0.032163743, 0.034188034, 0.036111111, 0.037940379, 0.03968254, 0.041343669, 0.042929293, 0.044444444],
            [0.057057057, 0.058479532, 0.05982906, 0.061111111, 0.062330623, 0.063492063, 0.064599483, 0.065656566, 0.066666667],
            [0.084084084, 0.084795322, 0.085470085, 0.086111111, 0.086720867, 0.087301587, 0.087855297, 0.088383838, 0.088888889],
            [0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111],
            [0.138138138, 0.137426901, 0.136752137, 0.136111111, 0.135501355, 0.134920635, 0.134366925, 0.133838384, 0.133333333],
            [0.165165165, 0.16374269, 0.162393162, 0.161111111, 0.159891599, 0.158730159, 0.157622739, 0.156565657, 0.155555556],
            [0.192192192, 0.19005848, 0.188034188, 0.186111111, 0.184281843, 0.182539683, 0.180878553, 0.179292929, 0.177777778],
            [0.219219219, 0.216374269, 0.213675214, 0.211111111, 0.208672087, 0.206349206, 0.204134367, 0.202020202, 0.2]]
        expected = np.round(lil_matrix(b).tocsr(), 6)
        a = self.makeTestMatrix()
        result = np.round(self.GraphMng.ColumnNomalization(a).tocsr(), 6)
        np.testing.assert_equal(result.toarray(), expected.toarray())
        # self.assertTrue((result.tocsr() != expected.tocsr())==0)

    def test_RowNomalization(self):
        b = [[0.022222222, 0.044444444, 0.066666667, 0.088888889, 0.111111111, 0.133333333, 0.155555556, 0.177777778, 0.2],
            [0.079365079, 0.087301587, 0.095238095, 0.103174603, 0.111111111, 0.119047619, 0.126984127, 0.134920635, 0.142857143],
            [0.09178744, 0.096618357, 0.101449275, 0.106280193, 0.111111111, 0.115942029, 0.120772947, 0.125603865, 0.130434783],
            [0.097222222, 0.100694444, 0.104166667, 0.107638889, 0.111111111, 0.114583333, 0.118055556, 0.121527778, 0.125],
            [0.100271003, 0.10298103, 0.105691057, 0.108401084, 0.111111111, 0.113821138, 0.116531165, 0.119241192, 0.12195122],
            [0.102222222, 0.104444444, 0.106666667, 0.108888889, 0.111111111, 0.113333333, 0.115555556, 0.117777778, 0.12],
            [0.103578154, 0.105461394, 0.107344633, 0.109227872, 0.111111111, 0.11299435, 0.114877589, 0.116760829, 0.118644068],
            [0.104575163, 0.10620915, 0.107843137, 0.109477124, 0.111111111, 0.112745098, 0.114379085, 0.116013072, 0.117647059],
            [0.105339105, 0.106782107, 0.108225108, 0.10966811, 0.111111111, 0.112554113, 0.113997114, 0.115440115, 0.116883117]]
        expected = np.round(lil_matrix(b).tocsr(), 6)
        a = self.makeTestMatrix()
        result = np.round(self.GraphMng.RowNomalization(a).tocsr(), 6)
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_QueryVec(self):
        expected = [[0.],
                    [0.1],
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                    [0.9],
                    [0.]]
        expected = lil_matrix(expected).tocsr()
        result = self.localTest.GraphMng.vecInitialState
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_matrixColNormTrans(self):
        b = [[0.2, 0, 0.266666667, 0.4, 0, 0.2, 0.2, 0.4, 0],
             [0, 0.2, 0, 0, 0.2, 0.2, 0.2, 0, 0.4],
             [0.16, 0, 0.2, 0, 0, 0.2, 0.2, 0, 0],
             [0.16, 0, 0, 0.2, 0, 0, 0, 0.4, 0],
             [0, 0.2, 0, 0, 0.2, 0.2, 0.2, 0, 0.4],
             [0.16, 0.2, 0.266666667, 0, 0.2, 0.2, 0, 0, 0],
             [0.16, 0.2, 0.266666667, 0, 0.2, 0, 0.2, 0, 0],
             [0.16, 0, 0, 0.4, 0, 0, 0, 0.2, 0],
             [0, 0.2, 0, 0, 0.2, 0, 0, 0, 0.2]]
        expected = np.round(lil_matrix(b).tocsr(), 6)
        result = np.round(self.localTest.GraphMng.objInfRank.matrixColNormTrans, 6)
        np.testing.assert_equal(result.toarray(), expected.toarray())

    def test_ExecRankingProcessFinalScore(self):
        b = [[0.15971769],
             [0.064981874],
             [0.019984381],
             [0.135136567],
             [0.014981874],
             [0.026042638],
             [0.026042638],
             [0.544227476],
             [0.008884861]]
        expected = np.round(lil_matrix(b).tocsr(), 6)
        result = np.round(self.localTest.GraphMng.vecInfRankVertexScore, 6)
        np.testing.assert_equal(result.toarray(), expected.toarray())


if __name__ == '__main__':
    unittest.main()
