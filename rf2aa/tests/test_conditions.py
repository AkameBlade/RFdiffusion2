import functools
import torch
import pandas as pd
import numpy as np
import itertools
from collections import OrderedDict
from hydra import initialize, compose

import rf2aa as rf
from rf2aa.data.compose_dataset import set_data_loader_params
from rf2aa.data.data_loader import loader_pdb, loader_complex, loader_na_complex, \
                                    loader_dna_rna
from rf2aa.data.loaders.rcsb_loader import loader_sm_compl_assembly_single, loader_sm_compl_assembly
from rf2aa.trainer_new import trainer_factory
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.tests import make_deterministic as make_deterministic
from rf2aa.set_seed import seed_all

# examples used in testing
pdb_item = {'Unnamed: 0': 282822, 'CHAINID': '6zsc_IC', 'DEPOSITION': '2020-07-15', 'RESOLUTION': 3.5, 'HASH': '068157', 'CLUSTER': 4707, 'SEQUENCE': 'MLPAAARPLWGPCLGLRAAAFRLARRQVPCVCAVRHMRSSGHQRCEALAGAPLDNAPKEYPPKIQQLVQDIASLTLLEISDLNELLKKTLKIQDVGLVPMGGVMSGAVPAAAAQEAVEEDIPIAKERTHFTVRLTEAKPVDKVKLIKEIKNYIQGINLVQAKKLVESLPQEIKANVAKAEAEKIKAALEAVGGTVVLE', 'LEN_EXIST': 30, 'TAXID': '9606'}
compl_item = {'CHAINID': '6vet_D:6vet_C', 'DEPOSITION': '2020-01-02', 'RESOLUTION': 1.46, 'HASH': '015533_020710', 'CLUSTER': 1650, 'LENA:B': '22:21', 'TAXONOMY': '9606:9606', 'ASSM_A': 1, 'OP_A': 0, 'ASSM_B': 1, 'OP_B': 0, 'HETERO': 'HETERO', 'HASH_A': '015533', 'HASH_B': '020710', 'LEN': [22, 21], 'LEN_EXIST': 43}
na_compl_item = {'CHAINID': '3sjm_C:3sjm_B_0', 'DEPOSITION': '2011-06-21', 'RESOLUTION': 1.35, 'HASH': '029615', 'CLUSTER': 13318, 'LENA:B:C:D': '64:18', 'TOPAD?': False, 'LEN': [64, 18], 'LEN_EXIST': 82}
rna_item = {'CHAINID': '4olb_B_0', 'DEPOSITION': '2014-01-23', 'RESOLUTION': 2.9, 'CLUSTER': '149', 'LENA:B': '10', 'LEN': [10], 'LEN_EXIST': 10}
sm_compl_item = {'CHAINID': '4res_C', 'DEPOSITION': '2014-09-23', 'RESOLUTION': 3.408, 'HASH': '042467', 'CLUSTER': 22914, 'SEQUENCE': 'MAGLSTDDGGSPKGDVDPFYYDYETVRNGGLIFAALAFIVGLIIILSKRLRCGGKKHRPINEDEL', 'LEN_EXIST': 32, 'LIGAND': [('Q', '1001', '17F')], 'ASSEMBLY': 1, 'COVALENT': [], 'PROT_CHAIN': 'C', 'LIGXF': [('Q', 10)], 'PARTNERS': [('C', 2, 23, 3.4699244499206543, 'polypeptide(L)'), ([('K', '2002', 'CLR')], [('K', 4)], 4, 4.189112663269043, 'nonpoly'), ('A', 0, 0, 6.709344863891602, 'polypeptide(L)'), ('B', 1, 0, 13.354745864868164, 'polypeptide(L)'), ([('L', '2003', 'K')], [('L', 5)], 0, 28.212512969970703, 'nonpoly')], 'LIGATOMS': 54, 'LIGATOMS_RESOLVED': 19, 'SUBSET': 'organic', 'name': '4res_C_asm1_Q1001-17F'}
sm_compl_covale_item = {'CHAINID': '6bgn_E', 'DEPOSITION': '2017-10-29', 'RESOLUTION': 1.51, 'HASH': '088784', 'CLUSTER': 2477, 'SEQUENCE': 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASKV', 'LEN_EXIST': 58, 'LIGAND': [('Y', '101', '6Y5')], 'ASSEMBLY': 5, 'COVALENT': [(('E', '1', 'PRO', 'N'), ('Y', '101', '6Y5', 'C3'))], 'PROT_CHAIN': 'E', 'LIGXF': [('Y', 1)], 'PARTNERS': [('E', 0, 75, 1.3035255670547485, 'polypeptide(L)')], 'LIGATOMS': 9, 'LIGATOMS_RESOLVED': 9, 'SUBSET': 'covale'}
sm_compl_asmb_item = {'CHAINID': '4i7z_D', 'DEPOSITION': '2012-12-01', 'RESOLUTION': 2.803, 'HASH': '045638', 'CLUSTER': 20270, 'SEQUENCE': 'MAQFTESMDVPDMGRRQFMNLLAFGTVTGVALGALYPLVKYFIPPSGGAVGGGTTAKDKLGNNVKVSKFLESHNAGDRVLVQGLKGDPTYIVVESKEAIRDYGINAVCTHLGCVVPWNAAENKFKCPCHGSQYDETGKVIRGPAPLSLALCHATVQDDNIVLTPWTETDFRTGEKPWWV', 'LEN_EXIST': 38, 'LIGAND': [('X', '201', '1E2')], 'ASSEMBLY': 1, 'COVALENT': [], 'PROT_CHAIN': 'D', 'LIGXF': [('X', 46)], 'PARTNERS': [('D', 6, 50, 3.2743349075317383, 'polypeptide(L)'), ('B', 2, 46, 2.9014363288879395, 'polypeptide(L)'), ([('P', '308', 'UMQ')], [('P', 30)], 41, 2.2776474952697754, 'nonpoly'), ('C', 4, 12, 3.0980331897735596, 'polypeptide(L)'), ('A', 0, 0, 5.883709907531738, 'polypeptide(L)'), ([('O', '307', 'UMQ')], [('O', 28)], 0, 6.910125732421875, 'nonpoly'), ('E', 8, 0, 12.058629035949707, 'polypeptide(L)'), ('H', 14, 0, 14.072593688964844, 'polypeptide(L)'), ('F', 10, 0, 15.534253120422363, 'polypeptide(L)'), ([('N', '306', 'UMQ')], [('N', 27)], 0, 16.360061645507812, 'nonpoly'), ([('K', '303', 'HEM')], [('K', 20)], 0, 16.7199649810791, 'nonpoly'), ('A', 1, 0, 19.487838745117188, 'polypeptide(L)'), ([('M', '305', '8K6')], [('M', 24)], 0, 22.37641716003418, 'nonpoly'), ('B', 3, 0, 22.523012161254883, 'polypeptide(L)')], 'LIGATOMS': 23, 'LIGATOMS_RESOLVED': 23, 'SUBSET': 'asmb'}
benchmark196_item = {'Unnamed: 0': 489, 'CHAINID': '4y7y_BA', 'DEPOSITION': '2015-02-16', 'RESOLUTION': 2.4, 'HASH': '105747', 'CLUSTER': 10756, 'SEQUENCE': 'TSIMAVTFKDGVILGADSRTTTGAYIANRVTDKLTRVHDKIWCCRSGSAADTQAIADIVQYHLELYTSQYGTPSTETAASVFKELCYENKDNLTAGIIVAGYDDKNKGEVYTIPLGGSVHKLPYAIAGSGSTFIYGYCDKNFRENMSKEETVDFIKHSLSQAIKWDGSSGGVIRMVVLTAAGVERLIFYPDEYEQL', 'LEN_EXIST': 196, 'TAXID': ''}

# configurations to test
configs = ["rf2aa", "rf2_deep_layerdropout", "legacy_train", "rf2aa_legacy_refinement", "rf_with_gradients", "untied_p2p"]
datasets = {
  "pdb"            : (loader_pdb,                      pdb_item,             {"homo": {"CHAIN_A": pd.Series(dtype=np.float32)}}),
  "compl"          : (loader_complex,                  compl_item,           {}),
  "na_compl"       : (loader_na_complex,               na_compl_item,        {}),
  "rna"            : (loader_dna_rna,                  rna_item,             {}),
  "sm_compl"       : (loader_sm_compl_assembly_single, sm_compl_item,        {}),
  "sm_compl_covale": (loader_sm_compl_assembly_single, sm_compl_covale_item, {}),
  "sm_compl_asmb"  : (loader_sm_compl_assembly,        sm_compl_asmb_item,   {}),
}
benchmark_datasets = {
  "pdb196": (loader_pdb, benchmark196_item, {"homo": {"CHAIN_A": pd.Series(dtype=np.float32)}})
}
cfg_overrides = [
    "loader_params.p_msa_mask=0.0", 
    "loader_params.crop=100000",
    "loader_params.mintplt=0",
    "loader_params.maxtplt=2",
    "loader_params.maxcycle=1",
    "loader_params.maxlat=2",
    "loader_params.maxseq=5"
]


# set up data for regression tests
@functools.lru_cache
def _setup_data(config=None, overrides=None):
    if config is None:
        config="base"
    if overrides is None:
        overrides = cfg_overrides

    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config, overrides=cfg_overrides)
 
    cfgs = cfg
    loader_params = set_data_loader_params(loader_params=cfg.loader_params) 
    names = list(datasets.keys())
    loaders = [ datasets[i][0] for i in names]
    items = [ datasets[i][1] for i in names]
    loader_kwargs = [ datasets[i][2] for i in names]

    loader_params_list  = [loader_params] * len(datasets)
    cfgs_list  = [cfgs] * len(datasets)
    return dict(zip(names, zip(names, items, loader_params_list, cfgs_list, loaders, loader_kwargs)))

def setup_data(config=None, overrides=None):
    if config is not None: config = tuple(config)
    if overrides is not None: overrides = tuple(overrides)
    return _setup_data(config, overrides)

# set up models for regression tests
@functools.lru_cache
def setup_models(device="cpu"):
    models, chem_cfgs = [], []
    for config in configs:
        with initialize(version_base=None, config_path="../config/train"):
            cfg = compose(config_name=config, overrides=cfg_overrides)

            # initializing the model needs the chemical DB initialized.  Force a reload
            ChemData.reset()
            ChemData(cfg.chem_params)

            trainer = trainer_factory[cfg.experiment.trainer](cfg)
            seed_all()
            trainer.construct_model(device=device)
            models.append(trainer.model)
            chem_cfgs.append(cfg)
            trainer = None 

    return dict(zip(configs, (zip(configs, models, chem_cfgs))))

# set up data for benchmark tests
@functools.lru_cache
def setup_benchmark_data(config=None):
    if config is None:
        config="base"
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config, overrides=cfg_overrides)
 
    cfgs = cfg
    loader_params = set_data_loader_params(loader_params=cfg.loader_params) 
    names = list(benchmark_datasets.keys())
    loaders = [ benchmark_datasets[i][0] for i in names]
    items = [ benchmark_datasets[i][1] for i in names]
    loader_kwargs = [ benchmark_datasets[i][2] for i in names]

    loader_params_list  = [loader_params] * len(benchmark_datasets)
    cfgs_list  = [cfgs] * len(benchmark_datasets)
    return dict(zip(names, zip(names, items, loader_params_list, cfgs_list, loaders, loader_kwargs)))

# set up trainers for benchmark tests
@functools.lru_cache
def setup_benchmark_trainers(device="cpu"):
    trainers= []
    for config in configs:
        with initialize(version_base=None, config_path="../config/train"):
            cfg = compose(config_name=config, overrides=cfg_overrides)

            ChemData.reset()
            ChemData(cfg.chem_params)
            trainer = trainer_factory[cfg.experiment.trainer](cfg)
            seed_all()
            trainer.construct_model(device=device)
            trainers.append(trainer)

    return dict(zip(configs, trainers))

# set up job array for regression
@functools.lru_cache
def _setup_array(datasets, models, device="cpu"):
    test_data = setup_data()
    test_models = setup_models(device=device)
    test_data = [test_data[dataset] for dataset in datasets]
    test_models = [test_models[model] for model in models]
    return list(itertools.product(test_data, test_models))
def setup_array(datasets, models, device='cpu'):
    return _setup_array(tuple(datasets), tuple(models), device)

# set up job array for benchmarking
@functools.lru_cache
def _setup_benchmark_array(datasets, models, device="cpu"):
    test_data = setup_benchmark_data()
    test_trainers = setup_benchmark_trainers(device=device)
    test_data = [test_data[dataset] for dataset in datasets]
    test_trainers = [test_trainers[model] for model in models]
    return (
        list(itertools.product(test_data, test_trainers)),
        ['-'.join(x) for x in itertools.product(datasets, models)]
    )
def setup_benchmark_array(datasets, models, device="cpu"):
    return _setup_benchmark_array(tuple(datasets), tuple(models), device)

def random_param_init(model):
    seed_all()
    with torch.no_grad():
        fake_state_dict = OrderedDict()
        for name, param in model.model.named_parameters():
            fake_state_dict[name] = torch.randn_like(param) * 1e-2
        model.model.load_state_dict(fake_state_dict)
        model.shadow.load_state_dict(fake_state_dict)
    return model

def dataset_pickle_path(dataset_name):
    return f"{rf.projdir}/test_pickles/data/{dataset_name}_regression.pt"

def model_pickle_path(dataset_name, model_name):
    return f"{rf.projdir}/test_pickles/model/{model_name}_{dataset_name}_regression.pt"

def loss_pickle_path(dataset_name, model_name):
    return f"{rf.projdir}/test_pickles/loss/{model_name}_{dataset_name}_loss_regression.pt"

def config_pickle_path(config_name):
    return f"{rf.projdir}/test_pickles/config/{config_name}_regression.pt"
