# Python function to buffer a Pandas DataFrame so you
# can add rows to it iteratively and still read the
# dataframe

import pandas as pd


class DataStore():
    """Class for efficiently maintaining a Pandas dataframe of
    time-series data that is received in real-time (discrete time
    instants)

    Parameters
    ----------
    columns : Index or array-like
        Column labels as for a Pandas Dataframe.
    """

    def __init__(self, columns, index_start=0, buffer_size=100):
        self.buffer_size_ = buffer_size
        self.data_ = self.prepare_empty_block_(index_start, 
                                    index_start + buffer_size, columns)
        self.last_index_ = None
        self.next_index_ = index_start

    @property
    def buffer_size(self):
        return self.buffer_size_

    @property
    def data(self):
        if self.last_index_ is not None:
            return self.data_.loc[:self.last_index_]
        else:
            return self.data_.loc[[]]

    @property
    def last_index(self):
        return self.last_index_

    @property
    def next_index(self):
        return self.next_index_

    @staticmethod
    def prepare_empty_block_(start_index, next_index, columns):
        index = pd.RangeIndex(start_index, next_index)
        return pd.DataFrame(index=index, columns=columns)

    def append_row(self, values):
        """Append one row of data at the bottom of the dataframe.

        Unlike pd.DataFrame.append this method does not return the
        dataframe. It is amended 'in place'.

        Parameters
        ----------
        values : dict-like object of data to append. The keys of the
            dict must be valid column names. Any missing data will
            be substituted with default missing data values.
        """
        if not set(values).issubset(self.data_.columns):
            raise(ValueError("invalid data items provided"))
        assert(self.next_index_ not in self.data.index)
        self.data_.loc[self.next_index_] = values
        self.inc_index_()

    def inc_index_(self, n=1):
        self.next_index_ += n
        self.last_index_ = self.next_index_ - 1
        next_block = self.data_.shape[0]
        while self.next_index_ >= len(self.data_):
            start_row, next_block = next_block, next_block + self.buffer_size
            empty_block = self.prepare_empty_block_(start_row, next_block,
                                                    self.data_.columns)
            self.data_ = pd.concat([self.data_, empty_block])
        assert(not self.data_.index.duplicated().any())

    def append_rows(self, data):
        """Append multiple rows of data at the bottom of the
        dataframe.

        Unlike pd.DataFrame.append this method does not return the
        dataframe. It is amended 'in place'.

        Parameters
        ----------
        data : DataFrame or list of dict-like objects containing
            the row data to append. In the case of a DataFrame,
            the index is ignored and the values are appended to
            the data store. For a list of dicts, each dict
            represents one row of data. The keys of each dict
            must be valid column names. Any missing data will 
            be substituted with NaN values.
        """
        n_rows = len(data)
        start, stop = self.next_index_, self.next_index_ + n_rows
        data = pd.DataFrame(data, dtype='object').set_index(
                                        pd.RangeIndex(start, stop))
        self.inc_index_(n_rows)  # inc index first
        self.data_.loc[start:stop-1, :] = data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()


"""
Example:
buffer_size = 100  # adjust to your needs
columns = ["DateTime", "open", "high", "low", "close", "volume",
           "open_interest"]
"""