import unittest
import numpy as np
from exhaustive_teacher import window_status,allowed_by_shadow
class ScreeningTests(unittest.TestCase):
    def test_window_selection(self):
        obstacle=np.array([[False,True],[False,False]])
        self.assertEqual(window_status(np.array([[1,0],[2,1]]),obstacle),'all_lit')
        self.assertEqual(window_status(np.zeros((2,2)),obstacle),'all_dark')
        self.assertEqual(window_status(np.array([[0,0],[1,1]]),obstacle),'eligible')
    def test_positive_addition_screen(self):
        masks=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=bool)
        response=.3*masks[0]+.5*masks[1]
        dark=response==0
        self.assertTrue(allowed_by_shadow(masks[0],dark))
        self.assertTrue(allowed_by_shadow(masks[1],dark))
        self.assertFalse(allowed_by_shadow(masks[2],dark))
        self.assertTrue(allowed_by_shadow(np.zeros(3,dtype=bool),dark))
if __name__=='__main__': unittest.main()
