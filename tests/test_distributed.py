import numpy as np
from sfsm_orchestration.core import sfsm_select
from sfsm_orchestration.distributed import mapreduce_checksum

def test_mapreduce_matches_serial():
    rng=np.random.default_rng(4); scores=rng.uniform(.01,.99,size=(200,8)); priors=rng.uniform(.1,.9,size=(200,8))
    serial=sfsm_select(scores,priors)
    checksum,count,_=mapreduce_checksum(scores,priors,workers=2,repeats=1)
    assert count==scores.shape[0]
    assert checksum==int(serial.sum())
