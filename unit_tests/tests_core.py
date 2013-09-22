import unittest2 as unittest
import vsm

from vsm import *

class TestCore(unittest.TestCase):

    def test_arr_add_field(self):

        arr = np.array([(1, '1'), (2, '2'), (3, '3')],
                   dtype=[('i', np.int), ('c', '|S1')])
        new_arr = np.array([(1, '1', 0), (2, '2', 0), (3, '3', 0)],
                       dtype=[('i', np.int), ('c', '|S1'), ('new', np.int)])

        new_field = 'new'
        vals = np.zeros(3, dtype=np.int)

        test_arr = arr_add_field(arr, new_field, vals)

        self.assertTrue((new_arr==test_arr).all())
        self.assertTrue(new_arr.dtype==test_arr.dtype)

    # def test_enum_matrix(self):


    def test_enum_sort(self):
        
        arr = np.array([7,3,1,8,2])
        sorted_arr = enum_sort(arr)
        sorted_arr1 = enum_sort(arr, indices=[10,20,30,40,50])

        self.assertTrue((sorted_arr == 
            np.array([(3, 8), (0, 7), (1, 3), (4, 2), (2, 1)],
            dtype=[('i', '<i8'), ('value', '<i8')])).all())

        self.assertTrue((sorted_arr1 ==
            np.array([(40, 8), (10, 7), (20, 3), (50, 2), (30, 1)], 
                  dtype=[('i', '<i8'), ('value', '<i8')])).all())


    def test_map_strarr(self):

        arr = np.array([(0, 1.), (1, 2.)], 
                   dtype=[('i', 'i4'), ('v', 'f4')])
        m = ['foo', 'bar']
        arr = map_strarr(arr, m, 'i', new_k='str')

        self.assertTrue((arr['str'] == np.array(m, 
                        dtype=np.array(m).dtype)).all())
        self.assertTrue((arr['v'] == np.array([1., 2.], dtype='f4')).all())


    def test_mp_split_ls(self):

        l = [slice(0,0), slice(0,0), slice(0,0)]
        self.assertTrue(len(mp_split_ls(l, 1)) == 1)
        self.assertTrue((mp_split_ls(l, 1)[0] == l).all())
        self.assertTrue(len(mp_split_ls(l, 2)) == 2)
        self.assertTrue((mp_split_ls(l, 2)[0] == 
                        [slice(0,0), slice(0,0)]).all())
        self.assertTrue((mp_split_ls(l, 2)[1] == [slice(0,0)]).all())
        self.assertTrue(len(mp_split_ls(l, 3)) == 3)
        self.assertTrue((mp_split_ls(l, 3)[0] == [slice(0,0)]).all())
        self.assertTrue((mp_split_ls(l, 3)[1] == [slice(0,0)]).all())
        self.assertTrue((mp_split_ls(l, 3)[2] == [slice(0,0)]).all())


suite = unittest.TestLoader().loadTestsFromTestCase(TestCore)
unittest.TextTestRunner(verbosity=2).run(suite)
