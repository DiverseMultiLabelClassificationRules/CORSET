import numpy as np
from scipy import sparse as sp
from mlrl.testbed import data
from sklearn.model_selection import train_test_split
from logzero import logger

from traitlets.config.configurable import Configurable
from traitlets import (
    Bool,
    Unicode,
    Int,
    List,
    Dict,
    Enum,
    Float,
    Instance,
    Integer,
    Undefined,
    Bool,
)

from .graph import construct_confidence_graph, convert_to_connectivity_graph
from .utils import flatten  # , filter_rows_with_no_labels
from .utils_data import read_data_arff, shuffle_split_data


def load_toy_dataset():
    X = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    Y = np.array([[1, 1], [0, 1], [1, 0]])
    trn_X, trn_Y, tst_X, tst_Y = (
        sp.csr_matrix(X),
        sp.csr_matrix(Y),
        sp.csr_matrix(X),
        sp.csr_matrix(Y),
    )
    feature_names = np.arange(X.shape[1])
    labels_names = np.arange(Y.shape[1])
    return trn_X, trn_Y, tst_X, tst_Y, feature_names, labels_names


class Dataset(Configurable):
    name = Unicode("", help="name of the dataset").tag(config=True)
    datadir = Unicode("./data", help="data directory").tag(config=True)
    split_train = Bool(False, help="split training data or not").tag(config=True)
    train_ratio = Float(0.7, help='ratio of training set to apply if split_train is True').tag(config=True)
    
    def load(self):
        if self.name == "toy":
            trn_X, trn_Y, tst_X, tst_Y, feature_names, labels_names = load_toy_dataset()

        elif (
            self.name == "birds"
            or self.name == "emotions"
            or self.name == "bookmarks"
            or self.name == "CAL500"
            or self.name == "mediamill"
        ):
            sparse_map = {
                "mediamill": False,
                "birds": False,
                "bookmarks": True,
                "CAL500": False,
                "emotions": False,
            }
            n_labels_map = {
                "mediamill": 101,
                "birds": 19,
                "bookmarks": 208,
                "CAL500": 174,
                "emotions": 6,
            }

            trn_X, trn_Y, tst_X, tst_Y, labels_names = read_data_arff(
                f"{self.datadir}/{self.name}",
                n_labels=n_labels_map[self.name],
                sparsity=0.9,
                train_test_split=0.7,
                sparse=sparse_map[self.name],
            )
            # convert to sparse
            trn_X = sp.lil_matrix(trn_X)
            trn_Y = sp.lil_matrix(trn_Y)
            tst_X = sp.lil_matrix(tst_X)
            tst_Y = sp.lil_matrix(tst_Y)

            feature_names = np.arange(trn_X.shape[1])  # TODO: this is a temporary fix, add the actual names

        elif self.name == "genbase":
            print("reading genbase")
            X, Y, md = data.load_data_set_and_meta_data(
                f"{self.datadir}/{self.name}", f"{self.name}.arff", f"{self.name}.xml"
            )
            trn_X, trn_Y, tst_X, tst_Y = shuffle_split_data(X, Y, 0.7, 1)
            labels_names = np.array([l.attribute_name for l in md.labels]) 
            feature_names = np.array([attr.attribute_name for attr in md.attributes])
        else:
            trn_X, trn_Y, md = data.load_data_set_and_meta_data(
                f"{self.datadir}/{self.name}",
                f"{self.name}-train.arff",
                f"{self.name}.xml",
            )
            tst_X, tst_Y, md = data.load_data_set_and_meta_data(
                f"{self.datadir}/{self.name}",
                f"{self.name}-test.arff",
                f"{self.name}.xml",
            )
            labels_names = np.array([l.attribute_name for l in md.labels])
            feature_names = np.array([attr.attribute_name for attr in md.attributes])
            
        # label id from/to name mapping
        self.label_names = labels_names
        self.label2id = {name: i for i, name in enumerate(self.label_names)}
        self.feature_names = feature_names
        self.feature2id = {name: i for i, name in enumerate(self.feature_names)}

        # label frequency
        self.label_freq = flatten(trn_Y.tocsc().sum(axis=0))

        self.trn_X, self.trn_Y, self.tst_X, self.tst_Y = trn_X, trn_Y, tst_X, tst_Y

        self._check_shapes()

        self.split_train_if_needed()

    def split_train_if_needed(self):
        if self.split_train:
            logger.info('split training set into {}/{}'.format(self.train_ratio, 1 - self.train_ratio))
            self.trn_X, self.dev_X, self.trn_Y, self.dev_Y = train_test_split(
                self.trn_X, self.trn_Y, train_size=self.train_ratio,
                random_state=1234
            )
        else:
            attrs_to_delete = ('dev_X', 'dev_Y')
            for attr in attrs_to_delete:
                if hasattr(self, attr):
                    delattr(self, attr)
            
    def build_confidence_graph(self, which="Y"):
        if which == "Y":
            mat, label_info = self.trn_Y, self.label_names
        elif which == "X":
            mat, label_info = self.trn_X, None

        self._g = construct_confidence_graph(mat.tocsc(), label_info)
        return self._g

    def build_connectivity_graph(self, which="Y"):
        return convert_to_connectivity_graph(self.build_confidence_graph(which))

    def set_data(self, trn_X, trn_Y, tst_X, tst_Y):
        self.trn_X, self.trn_Y, self.tst_X, self.tst_Y = trn_X, trn_Y, tst_X, tst_Y

        self.label_names = None
        self._check_shapes()
        self.split_train_if_needed()

    def _check_shapes(self):
        assert self.trn_X.shape[0] == self.trn_Y.shape[0], "{} != {}".format(
            self.trn_X.shape[0], self.trn_Y.shape[0]
        )
        assert self.tst_X.shape[0] == self.tst_Y.shape[0], "{} != {}".format(
            self.tst_X.shape[0], self.tst_Y.shape[0]
        )

        assert self.trn_X.shape[1] == self.tst_X.shape[1], "{} != {}".format(
            self.trn_X.shape[1], self.tst_X.shape[1]
        )
        assert self.trn_Y.shape[1] == self.tst_Y.shape[1], "{} != {}".format(
            self.trn_Y.shape[1], self.tst_Y.shape[1]
        )

    @property
    def ntrn(self):
        return self.trn_X.shape[0]

    @property
    def ndev(self):
        if self.split_train:
            return self.dev_X.shape[0]
        else:
            raise AttributeError('ndev not available because split_train is False')
    
    @property
    def ntst(self):
        return self.tst_X.shape[0]

    @property
    def ncls(self):
        return self.trn_Y.shape[1]

    @property
    def nfeat(self):
        return self.trn_X.shape[1]

    @property
    def shape(self):
        ret = dict(ntrn=self.ntrn, ntst=self.ntst, nfeat=self.nfeat, ncls=self.ncls)
        if self.split_train:
            ret['ndev'] = self.ndev
        return ret
