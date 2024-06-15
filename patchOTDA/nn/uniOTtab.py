## This script is a modified version of the original uniood script to work with tabular data, and/or to work with the patchOTDA framework
## The original script can be found at
## github.com/zhmiao/uniood/

import os

import torch
from .uniood.datasets.benchmark import MultiDomainsBenchmark, read_split_from_txt
from .uniood.methods import UniOT
from .uniood.engine.trainer import UniDaTrainer
from .uniood.engine.dataloader import build_data_loaders
from torch.utils.data import DataLoader, WeightedRandomSampler

from .uniood.datasets.utils import DatasetWrapper, FeatureWrapper
from .uniood.datasets.transforms import build_transform
from .uniood.datasets import dataset_classes
from collections import Counter
from umap import UMAP

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class defaultUNIOTcfg():
    model = "UniOT"
    method = "UniOT"
    max_iter = 1000

    #model config
    backbone = "FF"
    image_augmentation = "none"
    fixed_backbone = True
    fixed_BN = False
    ft_last_layer = False
    ft_norm_only = False
    ft_last_layer = False
    save_checkpoint = False
    eval_only = False
    save_checkpoint = False
    device = "cpu"
    backbone_output_dim = 2
    #optimizer / training
    optimizer = "sgd"
    base_lr = 1e-3
    weight_decay = 5e-4
    no_balanced = True
    num_workers = 0
    batch_size = 1024
    backbone_multiplier = 0.1
    momentum = 0.9
    clip_norm_value = 0
    lr_scheduler = "cosine"
    warmup_iter= 20
    warmup_type = "linear"
    warmup_min_lr = 1e-4
    
    #data params
    data_dir = "data"
    feature_dir = "features"
    dataset = "TabularDataset"
    use_features = True

DEFAULT_CFG = defaultUNIOTcfg()

class TabularDataset(MultiDomainsBenchmark):
    dataset_name = 'TabularDataset'
    #in this case, the domains are passed as numpy arrays, csv files, or pandas dataframes

    def __init__(self, source_domain, target_domain, source_labels=None, target_labels=None, label_col=None, n_share=None, n_source_private=None):
        if isinstance(source_domain, str):
            source_domain = pd.read_csv(source_domain)
        if isinstance(target_domain, str):
            target_domain = pd.read_csv(target_domain)

        #pull out the labels
        if label_col is not None:
            source_labels = source_domain[label_col].values
            target_labels = target_domain[label_col].values
            #encode them if they are strings
            if source_labels.dtype == 'object':
                source_labels, mapping_s= pd.factorize(source_labels)
            if target_labels.dtype == 'object':
                target_labels, mapping_t = pd.factorize(target_labels)
            #create the lab2cname mapping
            self.lab2cname = {i: mapping[i] for i in range(max(source_labels.max(), target_labels.max())+1)}
        else:
            #create the lab2cname mapping
            self.lab2cname = {i: str(i) for i in range(max(source_labels.max(), target_labels.max())+1)}

        #if source and target domains are pandas dataframes, we can use the following code to convert them to numpy arrays
        if isinstance(source_domain, pd.DataFrame):
            source_domain = source_domain.select_dtypes(include=['number']).to_numpy()
        #source_domain = source_domain.select_dtypes(include=['number']).to_numpy()
        if isinstance(target_domain, pd.DataFrame):
            target_domain = target_domain.select_dtypes(include=['number']).to_numpy()

        

        self.dataset_dir = ''
        self.source_path = ''#os.path.join(self.dataset_dir, self.domains[source_domain])
        self.target_path = '' #os.path.join(self.dataset_dir, self.domains[target_domain])
        source_total = self._numpy_to_dict(source_domain, source_labels)
        target_total = self._numpy_to_dict(target_domain, target_labels)

        self.domains = {'source': source_total, 'target': target_total}

        super().__init__(source_total=source_total, target_total=target_total, n_share=n_share, n_source_private=n_source_private)

  

    def _numpy_to_dict(self, X, Y=None):
        #dataloaders of Uniood will expect a list of dicts, where each dict has keys 'impath', 'label', and 'classname'
        #
        data = []
        for i in range(X.shape[0]):
            item = {'data': torch.tensor(X[i], dtype=torch.float32),
                    'impath': str(i),
                    'label': Y[i] if Y is not None else -1,
                    'classname': self.lab2cname[Y[i]] if Y is not None else ''}
            data.append(item)
        return data

    @staticmethod
    def from_numpy(Xs, Xt, Ys=None, Yt=None, n_share=None, n_source_private=None):
        return TabularDataset(source_domain=Xs, target_domain=Xt, source_labels=Ys, target_labels=Yt, n_share=n_share, n_source_private=n_source_private)
    

class uniOTtabModel(UniDaTrainer):
    def __init__(self, cfg=DEFAULT_CFG):
        #cheat and generate our config file
        super().__init__(cfg)
        self.source_data_loader, self.target_data_loader, \
                self.test_data_loader, self.val_data_loader = self.build_data_loaders(cfg)
        self.evaluator = self.build_evaluator(cfg)
        self.max_iter = cfg.max_iter
        self.cfg = cfg
        if self.use_features:
            self.model.backbone = torch.nn.Identity()
            if cfg.ft_last_layer:
                pass
        

    #the same but override the build_dataloader method
    def build_data_loaders(self, cfg):
        feature_dir = ''
        source_feature_path = ''
        target_feature_path = ''
        
        if (cfg.fixed_backbone or cfg.ft_last_layer) and os.path.exists(source_feature_path) and os.path.exists(target_feature_path):
            self.use_features = True
            print('Use pretrained features as dataloader')
            cfg.num_workers = 0
        else:
            self.use_features = False
            print('Use raw as dataloader')
            source_feature_path = None
            target_feature_path = None

        return build_data_loaders_TAB(cfg.dataset_raw, 
                                cfg.image_augmentation,
                                cfg.backbone,
                                cfg.no_balanced,
                                cfg.batch_size,
                                cfg.num_workers,
                                source_feature_path=source_feature_path,
                                target_feature_path=target_feature_path,
                                test_feature_path=target_feature_path,
                                val_feature_path=source_feature_path)
    

class UniOTtab(object):
    def __init__(self, cfg=DEFAULT_CFG, kwargs=None):
        #cfg is a default configuration object
        self.cfg = cfg #default configuration
        #update with kwargs
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(cfg, key, value)
        self.internal_classifier = RandomForestClassifier(n_estimators=500)
        self.internal_embedder = UMAP(n_components=2, n_neighbors=25)
        self.model = None

    
    def fit(self,Xs, Xt, Ys, Yt, n_share=None, n_source_private=None, n_target_private=None, num_internal_cluster=150, memory_len=2000, mu=0.7, cfg=DEFAULT_CFG, **kwargs):
        #Xs, Xt, Ys, Yt are numpy arrays
        #update cfg with n_share, n_source_private, n_target_private
        cfg.n_share = n_share if n_share is not None else cfg.n_share
        cfg.n_source_private = n_source_private if n_source_private is not None else cfg.n_source_private

        #update cfg with kwargs
        for key, value in kwargs.items():
            setattr(cfg, key, value)

        #update the backbone with the correct number of input features
        cfg.backbone = f"{cfg.backbone}_{Xs.shape[1]}_{cfg.backbone_output_dim}"
        dataset = TabularDataset.from_numpy(Xs, Xt, Ys, Yt, cfg.n_share, cfg.n_source_private)

        #cfg.dataset = dataset
        cfg.dataset_raw = dataset
        cfg.batch_size = min(cfg.batch_size, min((len(Ys)//2, len(Yt)//2)))

        #update the model hyperparameters
        UniOT._hyparas['num_cluster']['TabularDataset'] = num_internal_cluster
        UniOT._hyparas['memory_length']['TabularDataset'] = memory_len
        UniOT._hyparas['mu']['TabularDataset'] = mu
        UniOT._hyparas['feat_dim'] = 64
        UniOT._hyparas['lam'] = 0.5
        UniOT._hyparas['temperature'] = 0.1

        model = uniOTtabModel(cfg)
        model.train(dataset)
        self.model = model

        #fit the internal classifier
        self.internal_classifier.fit(self._predict(Xs)['features'].detach().numpy(), Ys)
        #fit the internal embedder
        self.internal_embedder.fit(self._predict(Xs)['features'].detach().numpy())
        return model
    
    def predict(self, X):
        #call internal predict method
        dict_result = self._predict(X)
        #predict the labels using the internal classifier
        return self.internal_classifier.predict(dict_result['features'].detach().numpy())
    
    def transform(self, X):
        #call internal predict method
        dict_result = self._predict(X)
        return self.internal_embedder.transform(dict_result['features'])

    def _predict(self, X):
        #no predict method for this model, labels are predicted during training
        return self.model.model.predict(dict(test_images=torch.tensor(X, dtype=torch.float32)))
    

def build_data_loaders_TAB(dataset, 
                       image_augmentation,
                       backbone,
                       no_balanced,
                       batch_size,
                       num_workers,
                       source_feature_path=None,
                       target_feature_path=None,
                       test_feature_path=None,
                       val_feature_path=None):
    data = dataset
    transform = build_transform(image_augmentation, backbone)
    sampler = None
    if not no_balanced:
        freq = Counter(data.train_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in data.train_labels]
        sampler = WeightedRandomSampler(source_weights, len(data.train_labels))

    source_loader = DataLoader(
                DatasetWrapper(data.train, transform=transform) if source_feature_path is None else FeatureWrapper(data.train, source_feature_path),
                batch_size=batch_size,
                sampler=sampler,
                shuffle=True if sampler is None else None,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
            )
    
    target_loader = DataLoader(
                DatasetWrapper(data.test, transform=transform) if target_feature_path is None else FeatureWrapper(data.test, target_feature_path),
                batch_size=batch_size,
                sampler=None,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
            )
    
    test_loader = DataLoader(
                DatasetWrapper(data.test, transform=build_transform("none", backbone)) if test_feature_path is None else FeatureWrapper(data.test, test_feature_path),
                batch_size=batch_size,
                sampler=None,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
            )
    
    val_loader = None
    if data.val is not None:
        val_loader = DataLoader(
                    DatasetWrapper(data.val, transform=build_transform("none", backbone)) if val_feature_path is None else FeatureWrapper(data.val, val_feature_path),
                    batch_size=batch_size,
                    sampler=None,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=torch.cuda.is_available(),
                )
    else:
        val_loader = DataLoader(
                    DatasetWrapper(data.train, transform=build_transform("none", backbone)) if val_feature_path is None else FeatureWrapper(data.train, val_feature_path),
                    batch_size=batch_size,
                    sampler=None,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=torch.cuda.is_available(),
                )
    
    return source_loader, target_loader, test_loader, val_loader