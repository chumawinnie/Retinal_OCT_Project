import unittest
import numpy as np
from analyse import evaluate
class EvaluationTests(unittest.TestCase):
    def test_patient_oof(self):
        rng=np.random.default_rng(42)
        y=np.repeat([0,1],30); X=rng.normal(size=(60,8)); X[:,0]+=y*2
        p,folds,m=evaluate(X,y)
        self.assertEqual(len(p),60)
        self.assertEqual(set(folds),set(range(5)))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(m['auroc'],.6)
        self.assertEqual(sum(map(sum,m['confusion_matrix'])),60)
    def test_small_class_rejected(self):
        with self.assertRaises(ValueError): evaluate(np.ones((8,2)),np.array([0]*4+[1]*4))
if __name__=='__main__': unittest.main()
