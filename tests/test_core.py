import numpy as np
from sfsm_orchestration.core import BenchmarkConfig, accuracy, fsm_select, generate_agent_graph, sfsm_select

def test_terminal_output_is_actual_index():
    scores=np.array([[.1,.9,.2],[.7,.1,.2]])
    priors=np.full_like(scores,.5)
    for selected in (fsm_select(scores,.6),sfsm_select(scores,priors)):
        assert selected.shape==(2,)
        assert np.all((0<=selected)&(selected<scores.shape[1]))

def test_one_million_agent_configuration():
    cfg=BenchmarkConfig()
    assert cfg.total_agents==1_000_000

def test_generation_and_accuracy():
    cfg=BenchmarkConfig(components=100,agents_per_component=8,test_seeds=1,timing_repeats=1,workers=1)
    correct,scores,priors=generate_agent_graph(.4,np.random.default_rng(1),cfg)
    selected=sfsm_select(scores,priors)
    value=accuracy(correct,selected)
    assert 0<=value<=1
