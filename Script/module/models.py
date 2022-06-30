import os
import copy
import datetime
import shutil
import random
import numpy as np
import pandas as pd
import gc

from sklearn.model_selection import  KFold, ParameterGrid
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import dgl
import dgl.nn.pytorch as dglnn
import networkx as nx

import module.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation for classification
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)      
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

#### Graph Object ####
class Graph_from_GSEA():
    """
    Build pathway graphs using annotation in GSEA    
    """
    def __init__(self, gmt_path = None, 
                 gml_path = None,
                gene_convert_ensembl = None,
                gene_convert_entrez  = None,
                ):
        """
        gmt_path : gmt file path which contains gene set
        gml_path : edge information in the pathways
        gene_convert_ensembl : .chip file
        gene_convert_entrez : .chip file
        """
        self.symbol_to_id = {}
        self.id_to_symbol = {}

        self.convert_ensembl = self.load_convert_dict(gene_convert_ensembl)
        self.convert_entrez  = self.load_convert_dict(gene_convert_entrez)
        
        self.load_gene_set(path=gmt_path)
        self.load_gml(path=gml_path)
        self.load_graph()
        
        #self.graph_names = [name for name in self.pathway_list.index] # graph_name
        
    def load_convert_dict(self, path):
        """each file is .chip file downloaded from GSEA"""
        return pd.read_csv(path, sep="\t", index_col=0)
        
    def load_gene_set(self, path):
        from gsea_api.molecular_signatures_db import GeneSets
        self.pathways = GeneSets.from_gmt(path)
        self.pathway_list = sorted([k for k in self.pathways.gene_sets_by_name])
        
    def load_gml(self, path):
        g = nx.read_graphml(path)
        # Rename
        mapping = {k: v for k, v in zip(g.nodes, [g.nodes.data()[node]["name"] for node in g.nodes])}
        self.gml = nx.relabel_nodes(g, mapping).to_undirected()
            
    def construct_graph(self, gene_set_name):
        """return dgl graph
        gene_set_name: GSEA KEGG gene set name, str:
           ex. "KEGG_MAPK_SIGNALING_PATHWAY"
        """
        # Nodes(genes) included in the pathway
        pathway_genes = sorted(list(self.pathways.gene_sets_by_name[gene_set_name].genes))
        symbol_to_id = {pathway_genes[i]: i for i in range(len(pathway_genes))}
        id_to_symbol = {i: pathway_genes[i] for i in range(len(pathway_genes))}
        self.symbol_to_id[gene_set_name] = symbol_to_id
        self.id_to_symbol[gene_set_name] = id_to_symbol

        # Get Subgraph from networkx
        nx_subgraph = nx.subgraph(self.gml, pathway_genes)
        edges = nx_subgraph.edges()
        src = [symbol_to_id[e[0]] for e in edges]
        dst = [symbol_to_id[e[1]] for e in edges]

        # Add nodes to Graph
        G = dgl.DGLGraph()
        G.add_nodes(len(pathway_genes))
        G.add_edges(src, dst)
        G.add_edges(dst, src)
        G.add_edges(G.nodes(), G.nodes())
        return G
    
    def load_graph(self):
        """load all graphs, use after func load_kgml()"""
        self.graphs = []
        for gene_set in self.pathway_list:
            self.graphs.append(self.construct_graph(gene_set))
        print("Graph loaded")

#### Load Omics Dataset ####
class Load_Dataset():
    def __init__(self, 
                 array_path  = None,
                 clinical_path = None,
                ):
        """Load omics data and clinical outcome dataset.
        array_path: str: array data file (pickle)
        clinical_path: str: clinical data file (csv)
        """
        # Array Data Loading
        # Expression
        self.x_expression = pd.read_pickle(array_path)
        
        # Load clinical data
        self.clinicalDf = self.load_clinical(clinical_path)

    def load_clinical(self, file):
        """load clinical data as pd.DataFrame
        clinical data is csv"""
        df = pd.read_csv(file, index_col=0)
        return df


#### Dataset for DGL ####
class Graph_Exp_Data(object):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self, 
                 x=None,
                 y=None, 
                 graphs=None, 
                 name=None,
                 url=None, 
                 force_reload=False, 
                 verbose=False):
        
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
                
        self.symbol_to_id = graphs.symbol_to_id
        self.id_to_symbol = graphs.id_to_symbol
        self.selected_genes = sorted(list(graphs.pathways.all_genes))
        self.graph_names = graphs.pathway_list
        self.graphs = graphs.graphs
        self.ngraphs = len(self.graphs)
        
        self.index = y.index
        self.load(x, y)
        self.process_graph()

    def download(self):
        # download raw data to local disk
        pass

    def __getitem__(self, idx):
        # get one example by index
        attrs = self.attr[idx, :]
        outcome = self.y[idx]
        return attrs, outcome

    def __len__(self):
        return self.x.shape[0]

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def trim_dataset(self, idx):
        import copy
        d = copy.deepcopy(self)
        idx_num = np.array([np.where(d.index==x)[0] for x in idx]).reshape(-1)
        _, d.y = d[idx_num]
        d.x = d.x.loc[idx, :]
        [g.ndata.clear() for g in d.graphs]
        d.attr = None
        d.index = np.array(idx, dtype="object")
        return d

    def load(self, x, y):
        # load processed data
        # load x, fill 0 if no value in selected gene
        nagenes = list(set(self.selected_genes) - set(x.columns))
        nagenesdf = pd.DataFrame([], index=x.index, columns=nagenes).fillna(0)
        self.x = x.join(nagenesdf)
        self.y = torch.from_numpy(y.values).reshape(-1).long()

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

    def process_graph(self):
        self.columns = []
        for i in range(self.ngraphs):
            d = self.id_to_symbol[self.graph_names[i]]
            self.columns = self.columns + [d[int(node)] for node in self.graphs[i].nodes()]
        self.batched_graph = dgl.batch(self.graphs)
        self.attr = torch.from_numpy(self.x.loc[:, self.columns].values)
        self.shapes = [self.attr.shape[1]]
        self.dims = 1

    def mapping_attr(self):
        self.attr = torch.from_numpy(self.x.loc[:, self.columns].values)
    
    def mapping_node_attr(self, X, id_hsa_dict, fill=0):
        node_names = list(id_hsa_dict.values())
        n = X.shape[0]
        df = pd.DataFrame([], columns=node_names, index=X.index)
        for i in range(len(node_names)):
            if node_names[i] in X.columns:
                df.iloc[:, i] = X[node_names[i]]
            else:
                df.iloc[:, i] = np.array([fill]*n)
        return df.astype(float)

    def map_all_nodes(self, X, attr_name="attr"):
        for i in range(len(self.graphs)):
            attr = self.mapping_node_attr(X, self.id_to_symbol[self.graph_names[i]])
            self.graphs[i].ndata[attr_name] = torch.from_numpy(attr.values.T).float()

#### Net ####
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim1=1000, hidden_dim2=1000, hidden_dim3=1000, 
                dropout=0, deep_fc=True, **kwargs):
        super().__init__()
        self.deep_fc = deep_fc

        if deep_fc:
            hidden_dims = [hidden_dim1, hidden_dim2, hidden_dim3]
            self.fc1 = nn.Linear(in_dim, hidden_dims[0])  
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
            self.final_dim = hidden_dims[2]
            
            self.norm1 = nn.BatchNorm1d(hidden_dims[0])
            self.norm2 = nn.BatchNorm1d(hidden_dims[1])
            self.norm3 = nn.BatchNorm1d(hidden_dims[2])

        else:
            self.final_dim = in_dim

class MLP_classifier(MLP):
    def __init__(self, in_dim, n_class, hidden_dim1=1000, hidden_dim2=1000, hidden_dim3=1000, 
                dropout=0, deep_fc=True, **kwargs):
        super().__init__(in_dim, hidden_dim1, hidden_dim2, hidden_dim3, 
                dropout, deep_fc)
        self.classify = nn.Linear(self.final_dim, n_class, bias=False)

    def forward(self, x):        
        if self.deep_fc:
            x = self.relu(self.fc1(x))
            x = self.norm1(x)
            x = self.dropout(x)

            x = self.relu(self.fc2(x))
            x = self.norm2(x)
            x = self.dropout(x)

            x = self.relu(self.fc3(x))
            x = self.norm3(x)
            x = self.dropout(x)
            x = self.classify(x)
            x = F.softmax(x, dim=1)
        return x

class GCN(nn.Module):
    def __init__(self, g, in_dim, gcn_dim1=10, gcn_dim2=10, hidden_dim1=0, hidden_dim2=0, 
    dropout=0.4, deep_fc=False, **kwargs):
        """
        Predict survival from multiomics / clinical data.
        GCN in DGL package is used.
        g: DGLgraph
        in_dim: int: dimention of each node in graph
        gcn_dim: int: gcn output dimension.
        hidden_dim: int: hidden layer dimension.
          if None, fc layer is omitted.
        dropout: float: dropout value
        deep_fc: bool: if true, combine fc layer after gcn.
        """
        super().__init__()
        self.gcn_dims = [gcn_dim1, gcn_dim2]
        self.hidden_dims = [hidden_dim1, hidden_dim2]

        self.in_dim = in_dim
        self.deep_fc = deep_fc

        # Layers
        self.g = copy.deepcopy(g)
        self.g.ndata.clear()
        self.ngraphs = g.batch_size
        self.num_nodes = g.number_of_nodes()
        self.conv1 = dglnn.GraphConv(self.in_dim, self.gcn_dims[0])
        self.conv2 = dglnn.GraphConv(self.gcn_dims[0], self.gcn_dims[1])
        self.hidden_dims = [self.gcn_dims[1] * self.ngraphs] + self.hidden_dims

        if deep_fc:
            self.fc1 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])  
            self.fc2 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
            self.final_dim = self.hidden_dims[2]
        else:
            self.final_dim = self.hidden_dims[0]
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        self.norm1 = nn.BatchNorm1d(self.hidden_dims[0])
        self.norm2 = nn.BatchNorm1d(self.hidden_dims[1])
        self.norm3 = nn.BatchNorm1d(self.hidden_dims[2])
        self.normg1 = nn.BatchNorm1d(self.gcn_dims[0])
        self.normg2 = nn.BatchNorm1d(self.gcn_dims[1])

    def graph_convolution(self, x):
        # Apply graph convolution and activation.
        x = self.relu(self.normg1(self.conv1(self.g, x)))
        x = self.relu(self.normg2(self.conv2(self.g, x)))

        # Calculate graph representation by average readout.
        with self.g.local_scope():
            self.g.ndata['x'] = x
            hg = dgl.mean_nodes(self.g, 'x')
            return torch.reshape(hg, (-1, ))

    def forward_gcn(self, x):
        self.n_batch = x.shape[0]
        x = torch.reshape(x, (self.n_batch, self.in_dim, self.num_nodes))
        x = torch.transpose(x, 1, 2)

        # Graph convolution
        x = torch.stack([self.graph_convolution(x_) for x_ in x], dim=0)
        x = self.norm1(x)
        x = self.dropout(x)
        return x

    def forward_fc(self, x):
        self.n_batch = x.shape[0]

        x = self.relu(self.fc1(x))
        x = self.norm2(x)
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.norm3(x)
        x = self.dropout(x)
        return x

class GCN_classifier(GCN):
    def __init__(self, g, in_dim, n_class, gcn_dim1=10, gcn_dim2=10, hidden_dim1=0, hidden_dim2=0, 
                 dropout=0.4, deep_fc=False, **kwargs):
        super().__init__(g, in_dim, gcn_dim1, gcn_dim2, hidden_dim1, hidden_dim2, 
                    dropout, deep_fc)

        self.classify = nn.Linear(self.final_dim, n_class, bias=False)
    
    def forward(self, x):
        x = self.forward_gcn(x)

        if self.deep_fc:
            x = self.forward_fc(x) 

        x = self.classify(x)
        x = F.softmax(x, dim=1)
        return x

class GCN_classifier_layer1(GCN_classifier):
    def __init__(self, g, in_dim, n_class, gcn_dim1=10, gcn_dim2=10, hidden_dim1=0, hidden_dim2=0, 
                dropout=0.4, deep_fc=False, **kwargs):
        super().__init__(g, in_dim, n_class, gcn_dim1, gcn_dim2, hidden_dim1, hidden_dim2, 
                         dropout, deep_fc)
    
    def forward(self, x):
        x = self.forward_gcn(x)
        return x

class GCN_classifier_layer2(GCN_classifier):
    def __init__(self, g, in_dim, n_class, gcn_dim1=10, gcn_dim2=10, hidden_dim1=0, hidden_dim2=0, 
                dropout=0.4, deep_fc=False, **kwargs):
        super().__init__(g, in_dim, n_class, gcn_dim1, gcn_dim2, hidden_dim1, hidden_dim2, 
                         dropout, deep_fc)
    
    def forward(self, x):
        if self.deep_fc:
            x = self.forward_fc(x) 
        x = self.classify(x)
        x = F.softmax(x, dim=1)
        return x

class ModelClassifier():
    def __init__(self, path="ModelRecord", device=None):
        self.path = path
        self.device = device

    def release_gpu_memory(self):
        try:
            del self.model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        return

    def create_weighted_sampler(self, Y):
        """
        Y: torch.tensor"""
        # Weighted sampler
        from torch.utils.data import WeightedRandomSampler
        Y = np.array(Y)
        Y = Y[torch.randperm(len(Y))]
        self.class_count = [i for i in utils.get_class_distribution(Y).values()]
        self.class_weights = 1/torch.tensor(self.class_count, dtype=torch.float)
        class_weights_all = self.class_weights[Y]
        self.weighted_sampler = WeightedRandomSampler(
                weights=class_weights_all,
                num_samples=len(class_weights_all),
                replacement=True
            )

    def build_network(self, in_dim, n_class, hidden_dim1, hidden_dim2, hidden_dim3, 
                    dropout=0, **kwargs):
        """build network"""
        pass

    def build_model(self, model_path=None, **kwargs):
        print("Build model.")
        self.build_network(**kwargs)
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            self.net.load_state_dict(state_dict, strict=True)
            print("Trained weight loaded.")
        return

    def load_bestmodel(self, load_log=True):
        # Load the saved best model
        if load_log:
            gs_scores_df = pd.read_csv(os.path.join(self.path, "grid_search.csv"), index_col=0)
        else:
            gs_scores_df = pd.DataFrame(self.log).T
        argmax = np.argmax(gs_scores_df['score'])
        p = gs_scores_df.iloc[argmax, :].to_dict()
        print(f'Best parameters: {p}')
        self.p = p
        self.build_model(model_path=os.path.join(self.path, "model.pth"), **self.p)

    def train(self, train_loader, val_loader, n_epoch, learning_rate=1e-5, EarlyStopping=True, patience=3, chkpt=None, **kwargs):
        self.accuracy_stats = {'train': [], "val": []}
        self.loss_stats = {'train': [], "val": [] }
        count = 0
        last_val_loss = 10e5
        min_val_loss  = 10e5
        self.n_epoch = n_epoch
        self.patience = patience
        self.EarlyStopping = EarlyStopping
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        self.optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)

        delete_tmp_dir = False
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d") + "_" + str(int(now.timestamp()))

        if chkpt is None:
            delete_tmp_dir = True
            TEMP_DIR  = os.path.join(self.path, "temp"+timestamp)
            os.makedirs(TEMP_DIR, exist_ok=True)
            self.chkpt = os.path.join(self.path, TEMP_DIR, "chkpt_"+timestamp+".pt")
        else:
            self.chkpt = chkpt

        print("Begin training.")
        for e in range(0, self.n_epoch):
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.net.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                self.optimizer.zero_grad()
                y_train_pred = self.net(X_train_batch)
                train_loss = self.criterion(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)
                train_loss.backward()
                self.optimizer.step()
                
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
                
            # VALIDATION    
            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0
                
                self.net.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    y_val_pred = self.net(X_val_batch)
                    val_loss = self.criterion(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    
            self.loss_stats['train'].append(train_epoch_loss/len(train_loader))
            self.loss_stats['val'].append(val_epoch_loss/len(val_loader))
            self.accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            self.accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                                    
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

            if self.EarlyStopping:
                if last_val_loss < val_epoch_loss:
                    count += 1
                    if count >= self.patience:
                        print("Early Stopping")
                        break
                    
            if self.chkpt is not None:
                if min_val_loss > val_epoch_loss:
                    min_val_loss = val_epoch_loss
                    torch.save(self.net.to("cpu").state_dict(), self.chkpt)
                    self.net = self.net.to(device)
                        
            last_val_loss = val_epoch_loss

        # load check point file before finish
        self.net.load_state_dict(torch.load(self.chkpt, map_location=torch.device("cpu")), strict=True)
        self.net.to(device)

        if delete_tmp_dir:
            shutil.rmtree(TEMP_DIR)

    def predict(self, X):
        with torch.no_grad():
            pred = self.net(X.to(device))
        return pred.to("cpu")

    def multi_acc(self, Y, pred):
        return multi_acc(pred, Y).numpy()

    def f1_score(self, Y, pred):
        return f1_score(Y, pred, average="binary")
    
    def roc_auc_score(self, Y, pred):
        return roc_auc_score(Y, pred[:, 1])

    def gridsearch_cv(self, X, Y, param_grid, cv=5, max_n_search=-1, random_seed=0, testscore=True,
    score_metrics="acc"):
        self.testscore = testscore
        n_iter           = 0
        score_dict       = {}
        max_score        = -1e10
        param_grid_list = list(ParameterGrid(param_grid))
        random.shuffle(param_grid_list)
        n_sample         = X.shape[0]

        if score_metrics == "acc":
            fn_score = self.multi_acc
        elif score_metrics == "f1_score":
            fn_score = self.f1_score
        elif score_metrics == "roc_auc":
            fn_score = self.roc_auc_score
            
        # Grid Search
        for p in param_grid_list:
            print(f"Hyper parameters: {p}")
            kf = KFold(n_splits = cv, shuffle=True, random_state=random_seed)
            i_cv = 0
            cv_score = []
            cv_max_score = 0
            cv_mean_score = 0
            for train_idx, val_idx in kf.split(range(n_sample)):
                self.release_gpu_memory()

                ######## Model training and evaluation #######
                X_train, X_val = X[train_idx,:], X[val_idx,:]
                Y_train, Y_val = Y[train_idx], Y[val_idx]

                # weighted sampler
                self.create_weighted_sampler(Y_train)

                # process dataset
                train_dataset = TensorDataset(X_train, Y_train)
                val_dataset = TensorDataset(X_val, Y_val)
                
                # dataloader
                train_loader = DataLoader(train_dataset, batch_size=p["n_batch"], sampler=self.weighted_sampler)
                val_loader = DataLoader(val_dataset, batch_size=p["n_batch"])

                self.build_model(**p)
                self.train(train_loader, val_loader, **p)
                pred = self.predict(X_val)
                #score = self.multi_acc(pred, Y_val)
                score = fn_score(Y_val, pred)
                print(f"Score: {score:.3g}")
                cv_score.append(score)

                # save parameters
                if score > cv_max_score:
                    cv_max_score = score
                    saved_parameters = self.net.to("cpu").state_dict()
                    self.net.to(device)

            # save scores
            cv_mean_score = np.mean(cv_score)
            print(f"CV mean score:{cv_mean_score:.3g}")
            score_dict[n_iter] = p
            score_dict[n_iter].update({"score":cv_mean_score})

            # Save model if the score was the highest
            if cv_mean_score > max_score:
                max_score = cv_mean_score
                print("saved best model")
                torch.save(saved_parameters, os.path.join(self.path, f"model.pth"))

            #### Test Set Evaluation ####
            if self.testscore:
                self.net.load_state_dict(saved_parameters, strict=True)
                # Test set
                print("Test set 1")
                pred = self.predict(self.X_test)
                score = fn_score(self.Y_test, pred)
                score_dict[n_iter].update({"test_set_score":score})
                print(f"test set score: {score:.3g}")

            #### End ####

            # record scores in csv
            self.log = score_dict
            gs_scores_df = pd.DataFrame(score_dict).T
            gs_scores_df.to_csv(os.path.join(self.path, "grid_search.csv"))

            n_iter += 1
            if n_iter == max_n_search:
                break

        print(f"Best score: {max_score:.3g}")
        print("Finish hyperparameter search.")


class MLP_Trainer(ModelClassifier):
    def __init__(self, path="ModelRecord", device=None):
        super().__init__(path, device)
    
    def build_network(self, in_dim, n_class, hidden_dim1, hidden_dim2, hidden_dim3, 
                    dropout=0, **kwargs):
        """build network"""
        self.net = MLP_classifier(int(in_dim), int(n_class), int(hidden_dim1), int(hidden_dim2), int(hidden_dim3), float(dropout), **kwargs)
        self.net = self.net.to(self.device)
        print(self.net)

class GCN_Trainer(ModelClassifier):
    def __init__(self, g, path="ModelRecord", device=None):
        super().__init__(path, device)
        self.g = g
    
    def build_network(self, in_dim, n_class, gcn_dim1=10, gcn_dim2=10, 
                    hidden_dim1=0, hidden_dim2=0, 
                    dropout=0, **kwargs):
        """build network"""
        self.net = GCN_classifier(self.g, int(in_dim), int(n_class), int(gcn_dim1), int(gcn_dim2), 
                    int(hidden_dim1), int(hidden_dim2), float(dropout), **kwargs)
        self.net = self.net.to(self.device) 
        print(self.net)   