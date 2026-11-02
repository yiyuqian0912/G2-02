"""RIND3D observation access; native physics is an explicit optional import."""

from .data import RIND3D, Window, TeacherScene, make_dataloader

__all__ = ["RIND3D", "Window", "TeacherScene", "make_dataloader"]
