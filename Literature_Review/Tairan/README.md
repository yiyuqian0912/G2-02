# Project Understanding

Based on the initial proposal, the project studies an inverse problem where the scene geometry is known but the illumination is only partially observed. The model takes the scene structure *S*, a local illumination observation *I_obs*, and the corresponding observation mask *M_R*, and aims to infer hidden light sources, reconstruct global illumination, and identify geometry responsible for blocking illumination.

The project is not only about recovering light sources. A further goal is to derive meaningful visibility structures from the reconstructed illumination field, including visibility-changing surfaces. Therefore, the broader goal is to learn a representation connecting partial observations, hidden physical causes, illumination, and geometric visibility. The proposal also evaluates whether these learned representations transfer to unseen scene geometries.

The key question is: **Can the learned representation capture underlying spatial and physical structure well enough to support inference and transfer to new geometries?**

# What I Think We Should Focus On

I think we should focus on whether the learned representation captures meaningful physical and geometric structure, and how to demonstrate this, rather than only improving prediction or reconstruction accuracy. Otherwise, our model would be just a run-of-the-mill supervised ML model, which conflicts with the goal of JEPA-RIND, as a so-called “world model.” In particular, we should examine whether the representation can identify visibility boundaries, explain illumination changes, and generalize to unseen scene geometries.

This also raises an important question about the role of the inverse model. We already have the JEPA-RIND forward model for learning representations and making predictions, but so far I do not see any clear relationship between the forward and inverse models. If the inverse model is completely separated from the forward model and its representation, then the claim that “inverse errors guide boundary refinement” is not supported by the architecture; it is a narrative rather than a mechanism.

Using a separate inverse model for exploration and testing is a legitimate engineering choice. However, in that case, we need to show that the inverse strategy provides information that cannot be obtained simply through forward prediction and active boundary refinement. Simply implementing and testing the inverse strategy does not answer this question.

I would be happy to develop such an ML model and I believe we could make it work well. But more fundamentally, we should ask whether we are building the inverse model because it is necessary for learning meaningful physical representations, or simply because we can make the inverse problem work.

# Paper Review 1: Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering

This paper combines a pretrained GAN with differentiable rendering to extract more explicit 3D structure from a learned latent representation. The main relevance to our project is the idea that a neural representation can be evaluated not only by reconstruction quality, but also by whether its latent variables correspond to meaningful physical properties.

For our project, this suggests testing whether the learned representation remains meaningful under different illumination conditions, observations, and scene geometries. However, reconstruction or consistency alone may not prove that the representation has truly learned the underlying physical structure.

# Paper Review 2: PhySG: Inverse Rendering with Spherical Gaussians for Physics-Based Material Editing and Relighting

PhySG uses a differentiable renderer to jointly reconstruct geometry, materials, and illumination from multi-view images. It represents geometry with an SDF and illumination and BRDFs with spherical Gaussians, allowing the reconstructed scene to be used for novel-view rendering, relighting, and material editing.

This is relevant because it shows how inverse rendering can combine learned representations with explicit physical constraints. For our project, the important idea is that a representation can be evaluated by whether it supports physically meaningful operations, rather than only matching the observed data.

# Paper Review 3: DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation

DeepSDF represents 3D shapes as continuous signed distance fields, where the zero level set defines the surface boundary and the sign indicates whether a point is inside or outside the shape. It can also learn a representation across a class of shapes and perform reconstruction and completion from partial data.

This is especially relevant to our project because our proposed event representation also aims to describe boundaries using a continuous function. DeepSDF provides a useful example of how a learned continuous field can encode meaningful geometric structure instead of directly predicting discrete boundary labels.

# Overall

Taken together, these papers suggest three aspects that may be particularly important for our project: meaningful latent representations, physical constraints, and explicit geometric structure. Image GANs shows the importance of testing whether a learned representation corresponds to interpretable physical properties. PhySG demonstrates how learned representations can be tied to physical rendering and evaluated through operations such as relighting. DeepSDF shows how a continuous learned field can explicitly represent geometric boundaries.

For our project, these ideas suggest that the main evaluation should go beyond reconstruction accuracy or inverse success. We should test whether the learned representation captures the underlying visibility and geometric structure, remains meaningful under changing conditions, and transfers to unseen geometries. At the same time, we should clarify whether the inverse model provides information that is genuinely useful for this goal, rather than simply providing another way to reconstruct the observed illumination.
