"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import logging
import os
import pickle
import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

smiles_to_num = {}


def load_data_from_df(dataset_path, add_dummy_node=True, one_hot_formal_charge=False, use_data_saving=True,
                      atomic_charge="", geom=False):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    feat_stamp = f'{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}{f"_{atomic_charge}" if atomic_charge else ""}{f"_{geom}" if geom else ""}'
    feature_path = dataset_path.replace('.csv', f'{feat_stamp}.pickle')
    if use_data_saving and os.path.exists(feature_path):
        logging.info(f"Loading features stored at '{feature_path}'")
        x_all, y_all = pickle.load(open(feature_path, "rb"))
        return x_all, y_all

    data_df = pd.read_csv(dataset_path)

    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    global smiles_to_num
    smiles_to_num = extract_smiles_to_num()

    x_all, y_all = load_data_from_smiles(data_x, data_y, add_dummy_node, one_hot_formal_charge, atomic_charge, geom)
    if use_data_saving and not os.path.exists(feature_path):
        logging.info(f"Saving features at '{feature_path}'")
        pickle.dump((x_all, y_all), open(feature_path, "wb"))

    return x_all, y_all


def load_data_from_smiles(x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=False,
                          atomic_charge="", geom=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all = [], []

    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)
            AllChem.ComputeGasteigerCharges(mol, throwOnParamFailure=True)

            afm, adj, dist = featurize_mol(mol, smiles, add_dummy_node, one_hot_formal_charge, atomic_charge, geom)
            x_all.append([afm, adj, dist, smiles])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all, y_all


def featurize_mol(mol, smiles, add_dummy_node, one_hot_formal_charge, atomic_charge="", geom=False):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge, atomic_charge)
                              for atom in mol.GetAtoms()])
    if atomic_charge in ["c", "n"]:  # Only for Chelpg, Npa; not for Gasteiger
        charge_feat = get_charge_from_file(atomic_charge, smiles, mol)
        node_features = np.c_[ node_features, charge_feat ]

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    # Allow for modified position matrix
    if not geom:
        pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                                for k in range(mol.GetNumAtoms())])
    else:
        pos_matrix = get_geom_from_file(geom, smiles, mol)
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge=True, atomic_charge=""):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    #TODO: Allow for modified atomic charge
    if atomic_charge == "g":
        attributes.append(atom.GetDoubleProp("_GasteigerCharge"))

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)

def extract_smiles_to_num():
    ans = {}
    with open("./data/Mathieu/Mathieu_logH_stratified.csv", "r") as f:
        count = 0
        for line in f.readlines():
            smi = line.split(",")[0]
            count += 1
            ans[smi] = count
    return ans

def get_geom_from_file(geom, smiles, mol):
    if geom == "hf":
        path = "hf_321G"
    elif geom == "b3lyp":
        path = "b3lyp_631Gdp"
    else:
        raise NotImplementedError(f"Geometry with theory of {geom} not implemented!")
    
    num = smiles_to_num[smiles]
    mol = Chem.AddHs(mol)
    logging.info(f"Getting geom data at ./data/Mathieu/gauss/{path}/geom/{num}.json")
    with open(f"./data/Mathieu/gauss/{path}/geom/{num}.json", "r") as f:
        mol_geom = json.load(f)

    ans = []
    for i in range(len(mol_geom)):
        assert mol.GetAtomWithIdx(i).GetAtomicNum() == mol_geom[i][0], f"Atomic num does not match for {smiles}"
        if mol_geom[i][0] != 1:
            ans.append(mol_geom[i][2:])
    
    return np.array(ans)

def get_charge_from_file(atomic_charge, smiles, mol):
    if atomic_charge == "c":
        path = "chelpg"
    elif atomic_charge == "n":
        path = "npa"
    else:
        raise NotImplementedError(f"Charge using theory of {atomic_charge} not implemented!")
    
    num = smiles_to_num[smiles]
    logging.info(f"Getting charge data at ./data/Mathieu/gauss/{path}/{num}.json")
    with open(f"./data/Mathieu/gauss/{path}/{num}.json", "r") as f:
        charge = json.load(f)

    ans = []
    for i in range(len(charge)):
        if charge[i][0] != "H":
            assert mol.GetAtomWithIdx(i).GetSymbol() == charge[i][0], f"Atomic num does not match for {smiles}"
            ans.append(charge[i][1])
    
    return np.array(ans)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.smiles = x[3]
        self.y = y
        self.index = index


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list = [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, labels)]


def construct_dataset(x_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], i)
              for i, data in enumerate(zip(x_all, y_all))]
    return MolDataset(output)


def construct_loader(x, y, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
                                         collate_fn=mol_collate_func,
                                         shuffle=shuffle)
    return loader
