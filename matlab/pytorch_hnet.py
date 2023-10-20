#!/usr/bin/python3
# A pytorch-based HNet inference engine; loads trained models and datasets from Export2JSON.m
# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
#   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
#   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
"""
CALLABLE FUNCTIONS:
    construct_hnet_model_from_json(filename:str, energy_mode:str) -> HNetModel
    evaluate(model:HNetModel, dataset:dict) -> np.ndarray
    main() -> None
"""
import os
import json
import re
import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
import sklearn.svm


# graph type
GRF_NULL            = 0 # n/a, null
GRF_GRID1D          = 1 # 1d grid
GRF_GRID2D          = 2 # 2d rectangular grid w 1 channel
GRF_GRID2DMULTICHAN = 3 # 2d rectangular grid w >1 channel
GRF_FULL            = 4 # fully connected
GRF_SELF            = 5 # each node is connected only to itself


# edge type
EDG_NULL  = 0 # n/a, null
EDG_T     = 1
EDG_NOR   = 2
EDG_NCONV = 3
EDG_NX    = 4
EDG_NIMPL = 5
EDG_NY    = 6
EDG_XOR   = 7
EDG_NAND  = 8
EDG_AND   = 9
EDG_NXOR  = 10
EDG_Y     = 11
EDG_IMPL  = 12
EDG_X     = 13
EDG_CONV  = 14
EDG_OR    = 15
EDG_F     = 16


UNIQ_EDGE_TYPES = [EDG_NULL,EDG_T,EDG_NOR,EDG_NCONV,EDG_NX,EDG_NIMPL,EDG_NY,EDG_XOR,EDG_NAND,EDG_AND,EDG_NXOR,EDG_Y,EDG_IMPL,EDG_X,EDG_CONV,EDG_OR,EDG_F]


# === HNET DATA STRUCTURES ===


class HNetEnergyViaHamiltonian(nn.Module):
    """energy matching via the Hamiltonian method"""
    def __init__(self, h:torch.Tensor, k:torch.Tensor):
        super().__init__()
        self.h:torch.Tensor = torch.Tensor(h) # n_cmp x n_nodes x n_nodes (Tensor) ...
        self.k:torch.Tensor = torch.Tensor(k) # n_cmp x 1 (Tensor) ...
        self.n_cmp:int      = int(self.h.shape[0]) # (int) number of components

    def forward(self, node_activations:torch.Tensor):
        """
        ...
        see also Energy.m
        
        Inputs
        ======
        node_activations - n_pts x n_nodes (Tensor) list of node activations for each datapoint

        Returns
        =======
        energies - n_pts x n_cmp (Tensor[double])
        """
        n_pts = node_activations.shape[0]
        energies = torch.zeros((n_pts,self.n_cmp), dtype=torch.double)
        for i in range(self.n_cmp):
            temp = torch.matmul(self.h[i,:,:], node_activations.T) # n_nodes x n_pts
            for j in range(n_pts):
                energies[j,i] = torch.dot(node_activations[j,:], temp[:,j])
            energies[:,i] += self.k[i]
        energies = torch.max(energies) - energies # convert from 0 = best to larger = better (similarity)
        return energies
    

class HNetEnergyViaEdgeMatching(nn.Module):
    """energy calculation via the edge matching method"""
    def __init__(self, learned_edge_states:torch.Tensor, edge_endnode_idx:torch.Tensor, edge_type_filter:torch.Tensor):
        super().__init__()
        self.learned_edge_states:torch.Tensor = torch.Tensor(learned_edge_states) # n_cmp x n_edges (Tensor[int64]) ...
        self.edge_endnode_idx:torch.Tensor    = torch.Tensor(edge_endnode_idx).to(torch.int64) # ...
        self.edge_type_filter:torch.Tensor    = torch.Tensor(edge_type_filter).to(torch.int64) # ...
        self.learned_edge_states_are_null:torch.Tensor = (self.learned_edge_states == EDG_NULL)# n_cmp x n_edges (Tensor[bool]) ...
        self.n_cmp:int = int(self.learned_edge_states.shape[0]) # (int) number of components
        
    def forward(self, node_activations:torch.Tensor):
        """
        see also Energy.m
        
        Inputs
        ======
        node_activations - n_pts x n_nodes (Tensor) list of node activations for each datapoint

        Returns
        =======
        energies - n_pts x n_cmp (Tensor[double])
        """
        n_pts = node_activations.shape[0]
        n_cmp = self.learned_edge_states.shape[0]
        edge_activations = _get_edge_states(node_activations, self.edge_endnode_idx, self.edge_type_filter)
        energies = torch.zeros((n_pts,self.n_cmp), dtype=torch.double)
        for i in range(n_pts):
            for j in range(n_cmp):
                temp = (self.learned_edge_states[j,:] == edge_activations[i,:])
                temp = torch.logical_or(temp, self.learned_edge_states_are_null[j,:])
                energies[i,j] = torch.sum(temp)
        energies = energies - torch.min(energies)
        return energies


class HNetEnergyViaBoolWeights(nn.Module):
    """energy calculation via the boolean weights method"""
    def __init__(self, learned_edge_states:torch.Tensor, edge_endnode_idx:torch.Tensor, edge_type_filter:torch.Tensor, do_include_null:bool, do_include_all_16:bool):
        super().__init__()
        self.edge_endnode_idx:torch.Tensor              = torch.Tensor(edge_endnode_idx).to(torch.int64) # ...
        self.edge_type_filter:torch.Tensor              = torch.Tensor(edge_type_filter).to(torch.int64) # ...
        self.binarized_learned_edge_states:torch.Tensor = torch.Tensor(_edge_to_logical(learned_edge_states, do_include_null, do_include_all_16)) # n_cmp x n_binarized_edges (Tensor) ...
        self.n_cmp:int                                  = int(self.binarized_learned_edge_states.shape[0]) # (int) number of components
        self.do_include_null:bool                       = bool(do_include_null) # ...
        self.do_include_all_16:bool                     = bool(do_include_all_16) # ...

    def forward(self, node_activations:torch.Tensor):
        """
        ...
        see also Energy.m
        
        Inputs
        ======
        node_activations - n_pts x n_binarized_edge_states (Tensor) list of node activations for each datapoint

        Returns
        =======
        energies - n_pts x n_cmp (Tensor[double])
        """
        n_pts = node_activations.shape[0]
        edge_activations = _get_edge_states(node_activations, self.edge_endnode_idx, self.edge_type_filter)
        binarized_edge_activations = _edge_to_logical(edge_activations, self.do_include_null, self.do_include_all_16)

        energies = torch.zeros((n_pts,self.n_cmp), dtype=torch.double)
        for i in range(n_pts):
            energies[i,:] = torch.matmul(self.binarized_learned_edge_states.to(torch.int8), binarized_edge_activations[i,:].to(torch.int8)) # implicit expansion

        energies = energies - torch.min(energies)
        return energies


class HNetNoNonlinearity(nn.Module):
    """module that performs no nonlinearity (returns input just as is)"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """x: n_pts x n_nodes (Tensor) list of energies for each datapoint"""
        premerge_idx = numpy.matlib.repmat(np.arange(x.shape[1]).T, 1, x.shape[1]) # n_groups x n_pts
        return x, premerge_idx


class HNetMax(nn.Module):
    """nonlinearity that finds the maximum input"""
    def __init__(self, learned_edge_states:torch.Tensor):
        super().__init__()
        self.learned_edge_states_not_null:torch.Tensor = torch.Tensor(learned_edge_states != EDG_NULL)
        
    def forward(self, x):
        """x: n_pts x n_nodes (Tensor) list of energies for each datapoint"""
        n_pts = x.shape[0]
        n_cmp = self.learned_edge_states_not_null.shape[1]
        newCompCode = np.zeros((n_pts,n_cmp), np.single)
        premerge_idx = np.zeros((n_pts,n_cmp))
        for i in range(n_cmp):
            idx = torch.nonzero(self.learned_edge_states_not_null[:,i])
            if idx.size > 0:
                newCompCode[:,i], premerge_idx[:,i] = torch.max(x[:,idx], dim=1)
                premerge_idx[:,i] = idx[premerge_idx[:,i]] # map back to the full list
        return newCompCode
    

class HNetMaxAbs(nn.Module):
    """nonlinearity that find the maximum absolute value input"""
    def __init__(self, learned_edge_states:torch.Tensor):
        super().__init__()
        self.learned_edge_states_not_null:torch.Tensor = torch.Tensor(learned_edge_states != EDG_NULL)
        
    def forward(self, x):
        """x: n_pts x n_nodes (Tensor) list of energies for each datapoint"""
        n_pts = x.shape[0]
        n_cmp = self.learned_edge_states_not_null.shape[1]
        newCompCode = torch.zeros((n_pts,n_cmp))
        premerge_idx = torch.zeros((n_pts,n_cmp))
        for i in range(n_cmp):
            idx = torch.nonzero(self.learned_edge_states_not_null[i,:])
            if idx.size > 0:
                newCompCode[:,i], premerge_idx[:,i] = torch.max(torch.abs(x[:,idx]), dim=1) # energy furthest from zero
                premerge_idx[:,i] = idx(premerge_idx[:,i]) # map back to the full list
        return newCompCode, premerge_idx


class HNetKWTA(nn.Module):
    """nonlinearity that performs k-winner-take-all"""
    def __init__(self, n_winners:int):
        super().__init__()
        self.n_winners:int = int(n_winners) # (int) number of winners
        
    def forward(self, x):
        """x: n_pts x n_nodes (Tensor) list of energies for each datapoint"""
        raise Exception("TODO")
        # x = k_wta(x, self.n_winners) # n_trn x n_cmp_groups
        # premerge_idx = ???
        # return x, premerge_idx


class HNetNonzero(nn.Module):
    """nonlinearity that returns 1 iff input is > 0"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """x: n_pts x n_nodes (Tensor) list of energies for each datapoint"""
        x = (x > 0)
        premerge_idx = numpy.matlib.repmat(np.arange(x.shape[1]).T, 1, x.shape[1]) # n_groups x n_pts
        return x, premerge_idx


class HNetComponentBank(nn.Module):
    """..."""
    def __init__(self, energy_mode:str, name:str, h:torch.Tensor, k:torch.Tensor, learned_edge_states:torch.Tensor, edge_endnode_idx:torch.Tensor, edge_type_filter:torch.Tensor, nonlinearity_mode:str, n_winners:int):
        super().__init__()
        do_include_null = False #TODO: should be exported from matlab
        do_include_all_16 = False #TODO: should be exported from matlab

        self.name:str = str(name) # just for printing
        self.n_cmp:int = int(h.shape[0]) # (int) number of components

        if energy_mode == "hamiltonian":
            self.energy = HNetEnergyViaHamiltonian(h, k)
        elif energy_mode == "edgematch":
            self.energy = HNetEnergyViaEdgeMatching(learned_edge_states, edge_endnode_idx, edge_type_filter)
        elif energy_mode == "boolweights":
            self.energy = HNetEnergyViaBoolWeights(learned_edge_states, edge_endnode_idx, edge_type_filter, do_include_null, do_include_all_16)
        else:
            raise Exception("unexpected energy mode")

        if nonlinearity_mode == "none":
            self.nonlinearity = HNetNoNonlinearity()
        elif nonlinearity_mode == "max":
            self.nonlinearity = HNetMax(learned_edge_states)
        elif nonlinearity_mode == "maxabs":
            self.nonlinearity = HNetMaxAbs(learned_edge_states)
        elif nonlinearity_mode == "kwta":
            self.nonlinearity = HNetKWTA(n_winners)
        elif nonlinearity_mode == "nonzero":
            self.nonlinearity = HNetNonzero()
        else:
            raise Exception("unexpected nonlinearity_mode")
        
    def forward(self, x):
        x = self.energy(x)
        x,_ = self.nonlinearity(x)
        return x


class HNetModel(nn.Module):
    """HNet Model"""
    def __init__(self, model_info:dict, energy_mode:str):
        super().__init__()
        self.links = model_info["links"]

        compbanks = [[]]*len(model_info["layout"])
        for i in range(len(model_info["layout"])): # for each component bank
            # convert dict into HNetComponentBank(nn.Module)
            name                = model_info["layout"][i]["name"]
            h                   = torch.Tensor(model_info["layout"][i]["h"])
            k                   = torch.Tensor(model_info["layout"][i]["k"])
            learned_edge_states = torch.Tensor(model_info["layout"][i]["learned_edge_states"]).to(torch.int64)
            edge_endnode_idx    = torch.Tensor(model_info["layout"][i]["edge_endnode_idx"]).to(torch.int64)
            edge_type_filter    = torch.Tensor(model_info["layout"][i]["edge_type_filter"]).to(torch.int64)
            nonlinearity_mode   = model_info["layout"][i]["nonlinearity_mode"]
            n_winners           = model_info["layout"][i]["n_winners"]
            compbanks[i] = HNetComponentBank(energy_mode, name, h, k, learned_edge_states, edge_endnode_idx, edge_type_filter, nonlinearity_mode, n_winners)

        if re.search("sense-->\w*,\w*-->out", self.links) is not None: # 1-tier architecture layout
            assert len(compbanks) == 1
            self.tier1 = compbanks[0]
        elif re.search("sense-->\w*,\w*-->\w*,\w*-->out", self.links) is not None: # 2-tier feed-forward architecture layout
            assert len(compbanks) == 2
            self.tier1 = compbanks[0]
            self.tier2 = compbanks[1]
        else:
            raise Exception("unexpected layout")

    def forward(self, x):
        if re.search("sense-->\w*,\w*-->out", self.links) is not None: # 1-tier architecture layout
            x = self.tier1(x)
        elif re.search("sense-->\w*,\w*-->\w*,\w*-->out", self.links) is not None: # 2-tier feed-forward architecture layout
            x = self.tier1(x)
            x = self.tier2(x)
        else:
            raise Exception("unexpected layout")
        
        return x


# === FILE-PRIVATE FUNCTIONS ===


def _remove_whitespace(text:str) -> str:
    return "".join(text.split())


def _load_dataset(filename:str):
    """
    ...

    Inputs
    ======
    filename - (str) full file name including path, for a file exported from training, ending in ".dataset.json" or similar
        format: {
            "comment": "<text notes>",
            "name": "<name of dataset>",
            "split": "<trn | tst>",
            "data": [<list of numbers in which the outermost (leftmost) dimension is the number of datapoints>],
            "label_idx": [<if the data comes with class labels, one per datapoint, those are here, else empty list>],
            <other metadata, depending on dataset>
        }
    """
    assert filename.endswith(".dataset.json")
    assert os.path.exists(filename), "file " + filename + " must exist"
    with open(filename, "r") as f:
        dataset = json.load(f)
    assert type(dataset["comment"]) is str
    assert type(dataset["name"]) is str
    assert type(dataset["split"]) is str
    data = np.array(dataset["data"], np.single)
    label_idx = np.array(dataset["label_idx"], np.single)
    return data, label_idx


def _filter_edge_type(edgestates:torch.Tensor, edge_type_filter:torch.Tensor):
    """
    filter edges, setting all to n/a except those listed in edgeTypeFilter
    modifies edgestates
    INPUTS
        edgestates - ? x n_edges (Tensor[int64]) ...
        edge_type_filter - ? x 1 (Tensor[int64]) ...
    """
    # e.g. with fully connected, there are SOOO many NCONV and NIMPL edges, so we just do AND
    if edge_type_filter.shape[0] > 0:
        mask = torch.zeros(edgestates.shape, dtype=torch.bool)
        for i in range(edge_type_filter.shape[0]):
            mask = mask | (edgestates == edge_type_filter[i])
        edgestates[torch.logical_not(mask)] = EDG_NULL


def _get_edge_states(data:torch.Tensor, didx:torch.Tensor, edge_type_filter:torch.Tensor) -> torch.Tensor:
    """
    ...

    Inputs
    ======
    data             - n x n_nodes (Tensor[bool]) node activations
    didx             - n_edges x 2 (Tensor[int64]) numeric index
    edge_type_filter - ? x 1 (Tensor[int64])
    
    Returns
    =======
    edgestates - n x n_edges (EDG enum)
    """
    n = data.shape[0]
    n_edges = didx.shape[0]
    assert torch.all((data == 0) | (data == 1))
    
    temp = torch.multiply(data[:,didx[:,0]], 2) + data[:,didx[:,1]]
    edgestates = torch.zeros((n,n_edges), dtype=torch.uint8)
    
    edgestates[temp == 0] = EDG_NOR
    edgestates[temp == 1] = EDG_NCONV
    edgestates[temp == 2] = EDG_NIMPL
    edgestates[temp == 3] = EDG_AND

    _filter_edge_type(edgestates, edge_type_filter)
    return edgestates


def _edge_to_logical(x:torch.Tensor, do_include_null:bool, do_include_all_16:bool) -> torch.Tensor:
    """
    ...

    Inputs
    ======
    x                 - n_pts x n_edges (categorical Tensor[int64]) ...
    do_include_null   - OPTIONAL scalar (bool) ...
    do_include_all_16 - OPTIONAL scalar (bool) ...

    Returns
    =======
    y - n_pts x n_edges*const (Tensor[bool])
    """
    n_pts, n_edges = x.shape

    if do_include_all_16 and do_include_null:
        n_edge_types = len(UNIQ_EDGE_TYPES)
    elif do_include_all_16 and not do_include_null:
        n_edge_types = len(UNIQ_EDGE_TYPES)-1
    elif not do_include_all_16 and do_include_null:
        n_edge_types = 5
    elif not do_include_all_16 and not do_include_null:
        n_edge_types = 4

    y = torch.zeros((n_edge_types,n_pts,n_edges), dtype=torch.bool)

    if do_include_all_16 and do_include_null:
        for i in range(len(UNIQ_EDGE_TYPES)):
            y[i,:,:] = (x == UNIQ_EDGE_TYPES[i])
    elif do_include_all_16 and not do_include_null:
        for i in range(1, len(UNIQ_EDGE_TYPES)): # skip EDG_NULL
            y[i-1,:,:] = (x == UNIQ_EDGE_TYPES[i])
    elif not do_include_all_16 and do_include_null:
        y[0,:,:] = (x == EDG_NULL)
        y[1,:,:] = (x == EDG_NOR)
        y[2,:,:] = (x == EDG_NCONV)
        y[3,:,:] = (x == EDG_NIMPL)
        y[4,:,:] = (x == EDG_AND)
    elif not do_include_all_16 and not do_include_null:
        y[0,:,:] = (x == EDG_NOR)
        y[1,:,:] = (x == EDG_NCONV)
        y[2,:,:] = (x == EDG_NIMPL)
        y[3,:,:] = (x == EDG_AND)
    
    y = torch.permute(y, [1,0,2]) # TODO correct permutation???
    y = torch.reshape(y, (n_pts,n_edge_types*n_edges))
    return y


# === CALLABLE FUNCTIONS ===


def construct_hnet_model_from_json(filename:str, energy_mode:str) -> HNetModel:
    """
    ...

    Inputs
    ======
    filename - (str) full file name including path, for a file exported from training, ending in ".hnetmodel.json"
        format:
        {
            "comment": "<text notes>",
            "links": "sense-->0,0-->1,1-->out", <where numbers index into the layout list>
            "layout": [
                {
                    "name": "<arbitrary component bank name>",
                    "h": [<list of numbers, n_cmp x n_nodes x n_nodes or empty list>],
                    "k": [<list of numbers, n_cmp x 1 or empty list>],
                    "learned_edge_states": [<list of numbers, ? x ? or empty list>],
                    "edge_endnode_idx": [...],
                    "edge_type_filter": [...],
                    "nonlinearity_mode": "<...>",
                    "n_winners": <int, ignored unless the layout calls for KWTA nonlinearity>
                }, { ... }, ...
            ]
        }
    energy_mode - (str) "hamiltonian" | "edgematch" | "boolweights"
    """
    assert filename.endswith(".hnetmodel.json")

    # load file to dict
    with open(filename, "r") as f:
        model_info = json.load(f)

    # validate file validity
    assert type(model_info["comment"]) is str
    assert type(model_info["layout"]) is list
    assert type(model_info["links"]) is str
    for compbank in model_info["layout"]:
        assert type(compbank["name"]) is str
        assert type(compbank["h"]) is list
        assert type(compbank["k"]) is list
        assert type(compbank["learned_edge_states"]) is list
        assert type(compbank["edge_endnode_idx"]) is list
        assert type(compbank["edge_type_filter"]) is list
        assert type(compbank["nonlinearity_mode"]) is str
        assert type(compbank["n_winners"]) is int

    # correct types and typos
    model_info["links"] = _remove_whitespace(model_info["links"].lower())
    for i in range(len(model_info["layout"])):
        model_info["layout"][i]["name"] = _remove_whitespace(model_info["layout"][i]["name"].lower())
        model_info["layout"][i]["nonlinearity_mode"] = _remove_whitespace(model_info["layout"][i]["nonlinearity_mode"].lower())

    # create & return nn.Module
    return HNetModel(model_info, energy_mode)


def evaluate(model:HNetModel, data) -> np.ndarray:
    """
    ...

    Inputs
    ======
    model - (HNetModel) ...
    data  - (ndarray|Tensor) ...

    Returns
    =======
    ...
    """
    torch.set_float32_matmul_precision("high")
    device = torch.device("cpu")
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    with torch.no_grad():
        output = model(torch.Tensor(data).to(device))
    output = output.cpu().numpy()
    return output


def main():
    """main function, if you just want to test things out (otherwise call evaluate from your own code)"""
    output_path = "D:\Projects/output_matlab/" # must end with "/"
    model_filename       = output_path + "basiccred_ucicreditgerman_tier1.memorize.hnetmodel.json"
    trn_dataset_filename = output_path + "ucicreditgerman_trn.dataset.json"
    tst_dataset_filename = output_path + "ucicreditgerman_tst.dataset.json"

    # model_filename       = output_path + "metacred_ucicreditgerman_tier1.memorize-tier1.extractcorr.icacropsome.100.50.unsupsplit-meta.extractcorr.kmeans.10.50.unsupsplit.hnetmodel.json"
    # trn_dataset_filename = output_path + "ucicreditgerman_trn.dataset.json"
    # tst_dataset_filename = output_path + "ucicreditgerman_tst.dataset.json"

    # model_filename       = output_path + "???.hnetmodel.json"
    # trn_dataset_filename = output_path + "mnistpy.128_trn.dataset.json"
    # tst_dataset_filename = output_path + "mnistpy.128_tst.dataset.json"

    trn_data, trn_label_idx = _load_dataset(trn_dataset_filename)
    tst_data, tst_label_idx = _load_dataset(tst_dataset_filename)

    model = construct_hnet_model_from_json(model_filename, "hamiltonian")
    output_trn = evaluate(model, trn_data)
    output_tst = evaluate(model, tst_data)

    model = construct_hnet_model_from_json(model_filename, "edgematch")
    output_tst2 = evaluate(model, tst_data)
    assert np.allclose(output_tst, output_tst2, 0.1)

    model = construct_hnet_model_from_json(model_filename, "boolweights")
    output_tst3 = evaluate(model, tst_data)
    # assert np.allclose(output_tst, output_tst3, 0.1) # TODO: test fails

    if tst_label_idx.size > 0:
        # raw SVM
        classifier = sklearn.svm.SVC()
        classifier.fit(trn_data, trn_label_idx)
        svm_pred = classifier.predict(tst_data)

        # SVM backend
        classifier = sklearn.svm.SVC()
        classifier.fit(output_trn, trn_label_idx)
        hnetsvm_pred = classifier.predict(output_tst)

        n_tst_datapoints = len(tst_label_idx)
        n_svm_correct = np.sum(svm_pred == tst_label_idx)
        n_hnetsvm_correct = np.sum(hnetsvm_pred == tst_label_idx)
        print("raw svm accuracy  = {}/{} ({:.0f}%)".format(n_svm_correct, n_tst_datapoints, 100. * n_svm_correct / n_tst_datapoints))
        print("hnet svm accuracy = {}/{} ({:.0f}%)".format(n_hnetsvm_correct, n_tst_datapoints, 100. * n_hnetsvm_correct / n_tst_datapoints))
    else:
        raise Exception("we currently only support datasets with categorical labels, one per datapoint")


if __name__=="__main__":
    main()