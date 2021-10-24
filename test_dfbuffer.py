# Tests of dfbuffer
import pandas as pd
import numpy as np
import unittest
from pandas.testing import assert_frame_equal
from dfbuffer import DataStore


class TestDataStore(unittest.TestCase):

    def test_append_row(self):

        # Test instantiation
        ds = DataStore(['a', 'b'])
        self.assertEqual(ds.next_index, 0)
        self.assertEqual(ds.buffer_size, 100)
        self.assertIsNone(ds.last_index)
        self.assertTrue(ds.data.empty)
        dfe = pd.DataFrame(None, index=range(100), columns=['a', 'b'], dtype='object')
        assert_frame_equal(ds.data_, dfe)
        test_str = "Empty DataFrame\nColumns: [a, b]\nIndex: []"
        self.assertEqual(str(ds), test_str)

        # Test append_row method
        ds.append_row({'a': 1, 'b': 2})
        dfe = pd.DataFrame([[1, 2]], index=[0], columns=['a', 'b'], dtype='object')
        assert_frame_equal(ds.data, dfe)
        ds.append_row({'b': 4})
        dfe = pd.DataFrame([[1, 2], [np.nan, 4]], index=[0, 1], columns=['a', 'b'],
                           dtype='object')
        assert_frame_equal(ds.data, dfe)
        ds.append_row({})
        dfe = pd.DataFrame([[1, 2], [np.nan, 4], [np.nan, np.nan]], index=[0, 1, 2],
                           columns=['a', 'b'], dtype='object')
        assert_frame_equal(ds.data, dfe)
        with self.assertRaises(ValueError):
            ds.append_row({'a': 7, 'c': 99})

        # Check __repr__ method
        self.assertEqual(
            str(ds),
            '     a    b\n0    1    2\n1  NaN    4\n2  NaN  NaN'
        )

        # Test other arguments
        ds = DataStore(columns=['a', 'b'], index_start=900, buffer_size=1000)
        self.assertIsNone(ds.last_index)
        self.assertEqual(ds.next_index, 900)
        self.assertEqual(ds.buffer_size, 1000)
        self.assertTrue(ds.data.empty)

    def test_append_rows(self):

        # Test adding multiple rows
        ds = DataStore(columns=[1, 2, 3])
        df = pd.DataFrame(np.arange(15).reshape((5, 3)), columns=[1, 2, 3])
        ds.append_rows(df)
        self.assertTrue(ds.data.equals(df.astype('object')))
        self.assertEqual(ds.last_index, 4)
        self.assertEqual(ds.next_index, 5)
        ds.append_row({1: 1, 2: 1, 3: 1})
        ds.append_row({1: 2, 2: 2, 3: 2})
        self.assertEqual(ds.last_index, 6)
        self.assertEqual(ds.next_index, 7)
        test_str = ("    1   2   3\n"
                    "0   0   1   2\n"
                    "1   3   4   5\n"
                    "2   6   7   8\n"
                    "3   9  10  11\n"
                    "4  12  13  14\n"
                    "5   1   1   1\n"
                    "6   2   2   2")
        self.assertEqual(str(ds), test_str)

    def test_buffering(self):

        n = np.arange(0, 1000, 10)
        data = [{'i': i, 'n': n, 'csum': csum} for i, (n, csum) in
                enumerate(np.vstack([n, n.cumsum()]).T)]

        # Big enough buffer
        ds1 = DataStore(columns=['i', 'n', 'csum'], buffer_size=101)
        ds1.append_rows(data)
        self.assertEqual(ds1.data_.shape, (101, 3))
        self.assertEqual(ds1.last_index, 99)
        self.assertEqual(ds1.next_index, 100)

        # Big enough buffer - but will trigger an extension
        ds2 = DataStore(columns=['i', 'n', 'csum'], buffer_size=100)
        ds2.append_rows(data)
        self.assertEqual(ds2.data_.shape, (200, 3))
        self.assertEqual(ds2.last_index, 99)
        self.assertEqual(ds2.next_index, 100)
        self.assertTrue(ds2.data.equals(ds1.data))

        # Too small buffer
        ds3 = DataStore(columns=['i', 'n', 'csum'], buffer_size=99)
        ds3.append_rows(data)
        self.assertEqual(ds3.data_.shape, (198, 3))
        self.assertEqual(ds3.last_index, 99)
        self.assertEqual(ds3.next_index, 100)
        self.assertTrue(ds3.data.equals(ds1.data))

        # Very small buffer
        ds4 = DataStore(columns=['i', 'n', 'csum'], buffer_size=10)
        ds4.append_rows(data)
        self.assertEqual(ds4.data_.shape, (110, 3))
        self.assertEqual(ds4.last_index, 99)
        self.assertEqual(ds4.next_index, 100)
        self.assertTrue(ds4.data.equals(ds1.data))

        # Incrementally add data of different sizes
        ds5 = DataStore(columns=['i', 'n', 'csum'], buffer_size=8)
        rng = np.random.RandomState(0)
        sample_sizes = [13, 4, 24, 1, 4, 15, 8, 10, 2, 18, 3]
        i = 0
        j = 0
        for _ in range(100):
            if rng.randint(2) == 0:
                # Add one row
                ds5.append_row(data[i])
                n = 1
            else:
                # Add multiple rows
                n = min(len(data), i + sample_sizes[j])
                ds5.append_rows(data[i:i+n])
                j += 1
            i += n
            if i >= len(data):
                break
            self.assertEqual(i, len(ds5))

        self.assertTrue(ds5.data.equals(ds1.data))
