import pytest
import torch
from hydra import initialize, compose
from rf2aa.metrics.metrics_factory import metrics_factory, MetricManager
from rf2aa.tests.test_conditions import setup_array, model_pickle_path, configs
from rf2aa.tests.test_model import setup_test

test_configs = [config for config in configs if "legacy" not in config]
gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize('conf', test_configs)
def test_metrics(conf):
    example, model = setup_array(['pdb'], [conf])[0]
    #seting up the test
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    model_pickle = model_pickle_path(dataset_name, model_name)
    rf_outputs = torch.load(model_pickle, map_location=gpu, weights_only=False)["outputs"]
    loss_calc_items = None
    for metric_name, metric in metrics_factory.items():
        #calling the function
        try:
            metric(rf_outputs, loss_calc_items)
        except Exception as e:
            raise ValueError(f"{metric_name} fails with following exception: {e}") from e

def test_metric_config():
    config = "base"
    cfg_overrides = []
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config, overrides=cfg_overrides)
    
    metrics_manager = MetricManager(cfg)

    assert len(metrics_manager.metrics) == 2
    assert "mean_pae" in metrics_manager.metrics
    assert "mean_plddt" in metrics_manager.metrics
