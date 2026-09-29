
from e2ec_config import *

from e2ec.workflow import Workflow
Workflow(
    device='cuda:0',
    modality=cfg.modality,
    dpath_vec=cfg.dpath_vec,
    fpath_FCStatedict=cfg.fpath_FCStatedict,
)

"""
when implementing a new method, dont
go in completely blind. try to have
an overview at least so that you know
things like what inputs and outputs
are expected and what they mean.

!keep this bord.

...
"""

"""
* figure out how to reverse the dissassembly
* continue deeplift vid
* ...
"""