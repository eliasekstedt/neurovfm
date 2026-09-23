
from e4dlft import *

from e4dlft.workflow import Workflow
Workflow(
    device=cfg.device,
    fpath_statedictEncoder=cfg.fpath_statedictEncoder,
    fpath_statedictFC=cfg.fpath_statedictFC,
)