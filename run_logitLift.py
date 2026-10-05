"""
likely reason deeplift maps appear scattered is mean pooling

if logit approach proves good enough you just spent a week on something
you did not need to do.

that is why it is worth thinking about what you are doing and why you are doing it
and make sure your current undertaking truly is the right path forward.
"""


from config_logitLift import *

if cfg.do_downFlow:
    from logitLift.downflow import DownFlow
    DownFlow(
        mriseq=cfg.mriseq,
        dpath_niiRoot=cfg.dpath_niiRoot,
        dpath_voxlvec=cfg.dpath_voxlvec,
    )

if cfg.do_encoderFlow:
    from logitLift.encoderflow import EncoderFlow
    EncoderFlow(
        device=cfg.device,
        dpath_voxlvec=cfg.dpath_voxlvec,
        dpath_embvec=cfg.dpath_embvec,
        fpath_encoderState=cfg.fpath_encoderState,
    )

if cfg.do_logitFlow:
    from logitLift.logitflow import LogitFlow
    LogitFlow(
        device=cfg.device,
        dpath_embvec=cfg.dpath_embvec,
        dpath_logitvec=cfg.dpath_logitvec,
        fpath_FCState=cfg.fpath_FCState,
    )

if cfg.do_upFlow:
    from logitLift.upflow import UpFlow
    UpFlow(
        buildkey=cfg.buildkey,
        dpath_logitvec=cfg.dpath_logitvec,
        dpath_rebuilt=cfg.dpath_rebuilt,
    )
#dpath_logitvec=cfg.dpath_logitvec,