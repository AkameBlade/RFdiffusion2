import pytest
import os
import torch
import itertools
from omegaconf import OmegaConf
from rf2aa.tests.test_conditions import dataset_pickle_path, config_pickle_path, loss_pickle_path, make_deterministic, random_param_init
from rf2aa.chemical import ChemicalData as ChemData

from rf2aa.trainer_new import trainer_factory

test_conditions = list(itertools.product(["pdb", "na_compl", "rna", "sm_compl", "sm_compl_covale"], ["rf2aa"]))

@pytest.mark.gpu
@pytest.mark.parametrize("dataset,model", test_conditions)
def test_loss_functions(dataset, model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_path = dataset_pickle_path(dataset)
    dataset_inputs = torch.load(dataset_path, map_location=torch.device(device), weights_only=False) 

    # load config pickle
    config_path = config_pickle_path(model)
    config = OmegaConf.create(torch.load(config_path, map_location="cpu", weights_only=False))

    trainer = trainer_factory[config.experiment.trainer](config)
    ChemData.reset()
    ChemData(config.chem_params)

    trainer.move_constants_to_device(device)
    trainer.construct_model(device)

    trainer.model = random_param_init(trainer.model)
    trainer.model.device = device
    make_deterministic()
    with torch.no_grad():
        loss, loss_dict = trainer.train_step(dataset_inputs, 1, device)
    
    loss_pickle_filepath = loss_pickle_path(dataset, model)
    if not os.path.exists(loss_pickle_filepath):
        torch.save(
            {
            "loss": loss, 
            "loss_dict": loss_dict
            }
        , loss_pickle_filepath)
    else:
        loss_dict_old = torch.load(loss_pickle_filepath, weights_only=False)
        for key in loss_dict.keys():
            # losses involving sidechains seem to be more inaccurate on balance
            atol = 1e-5 if key not in \
            ["torsion", "allatom_lddt", "allatom_lddt_prot_intra",
             "allatom_lddt_lig_intra", "allatom_lddt_prot_lig_inter", "allatom_fape", \
                "rmsd", "rmsd_prot_prot", "rmsd_prot_tgt", \
             "rmsd_prot_lig", "rmsd_lig_lig", "clash_loss"] \
                else 5e-1
            if key in ["bond_geom", "clash_loss", "rmsd", "rmsd_prot_tgt"] and dataset in ["rna", "na_compl"]:
                # bond_geom and clash loss is not deterministic for rna 
                continue
            if key == "total_loss": # test that weights are approximately applied correctly
                atol = 2.0
            #NOTE: rmsd_prot_lig seems to be unstable in some cases (probably since structures are bad and SVD solutions are different)
            # this is also the case for clash_loss
            if key in ["rmsd_prot_lig", "rmsd_prot_tgt", "clash_loss"]: # this overrides the previous atol setting, hoping to remove this and fallback to the previous setting
                atol = 2.0
            try:
                assert torch.allclose(loss_dict[key], loss_dict_old["loss_dict"][key], atol=atol)
            except Exception as e:
                raise AssertionError(f"Error in {dataset} {model} {key} {loss_dict[key]} {loss_dict_old['loss_dict'][key]}") from e