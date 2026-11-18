"""RIND3D authoritative voxel physics (Model B), research prototype (docs/VOXEL_PHYSICS_DESIGN.md).

Five objects are kept distinct:
1. generative continuous geometry (canonical polytope bundles, balls): inputs and reproducible metadata only;
2. the authoritative voxel solid G_V = union of the closed cells selected by a declared rule (`voxelize`);
3. Student-visible local observations (SOLID = cell occupancy at the window's cells; responses at cell centres);
4. the continuous feasible source domain (any legal point of the open cube, not voxel centres);
5. the certified global posterior (128^3 cell masses of the uniform law on the feasible set).
"""
VERSION = "rind3d-voxel-physics-v0"
