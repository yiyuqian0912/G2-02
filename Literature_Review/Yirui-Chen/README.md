# Literature Review

## Project Understanding

My understanding of this project is that the main goal is to learn the
underlying geometry of visibility-switching events rather than only predicting
discrete visible or occluded states. The model should capture how scene
structure and light parameters determine these events, and ideally transfer
these relationships to unseen structures. Another important part is inversion:
using observed shadows to infer possible light parameters and using physical
validation to check whether the learned event relationships are actually
correct.

## Possible Method

One possible approach is to first train an event model using ray-traced
visibility observations. The model can learn the relationship between scene
structure, query location, light direction, and visibility, while also
representing possible visibility-switching boundaries.

We can then use the learned model for shadow-to-light inversion. Given an
observed shadow, the model can generate candidate light directions, which can
be checked using an independent physical ray tracer. If a candidate fails the
physical validation, the failure may indicate that the model learned an event
boundary incorrectly. We can then add more ray-tracing queries around the
related region and use these new observations to refine the model.

Finally, we can test whether the learned event relationships transfer to
unseen structures. A useful experiment could compare direct visibility
prediction, general boundary-based sampling, and inversion-guided sampling
under the same ray-tracing query budget.

---

## 1. Occupancy Networks: Learning 3D Reconstruction in Function Space

**Mescheder et al. (2019), CVPR**

Occupancy Networks represent 3D geometry using a continuous neural function.
Instead of storing an object using a fixed voxel grid, the model predicts
whether a given point is inside or outside the object, and the object surface
is represented by the decision boundary of this continuous function.

This gives us a useful way to think about event representation. Similarly, we
could use a continuous function to represent visibility, where its zero-level
set corresponds to a visibility-switching boundary. This could allow the model
to learn not only visible or occluded states, but also information about where
the transition occurs.

---

## 2. NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis

**Srinivasan et al. (2021), CVPR**

NeRV uses neural fields to represent scene properties related to reflectance
and visibility. An important part of the model is its ability to predict
visibility from a spatial location toward different directions, showing that
directional visibility can be learned using a continuous neural
representation.

This is closely related to our visibility-learning problem. NeRV provides a
useful reference for learning the relationship between spatial locations,
directions, and visibility. Our project can extend this idea by focusing more
explicitly on the locations of visibility-switching events and testing whether
these event relationships can transfer to unseen structures.

---

## 3. An Adaptive Strategy for Active Learning with Smooth Decision Boundary

**Locatelli, Carpentier, and Kpotufe (2018), ALT**

This paper studies active learning around decision boundaries. Instead of
sampling the entire input space uniformly, the method places more queries in
regions where the decision boundary is uncertain. This allows a limited query
budget to be used more efficiently.

This idea is useful for our ray-tracing process because simulation queries can
be expensive. Rather than adding samples uniformly across the entire light
parameter space, we could focus additional queries around uncertain
visibility-switching regions. It also provides a useful comparison for testing
whether inversion-guided sampling is more effective than general
boundary-focused sampling.

---

## 4. Reparameterizing Discontinuous Integrands for Differentiable Rendering

**Loubet, Holzschuch, and Jakob (2019), ACM TOG**

This paper focuses on discontinuities in differentiable rendering. Visibility
and shadows can change suddenly when geometry or illumination changes, which
makes standard differentiation difficult. The authors introduce a
reparameterization method that allows useful gradients to be estimated even
when these visibility discontinuities are present.

This is closely related to the "Beyond Jacobian" motivation of our project.
For hard visibility, local derivatives may provide limited information about
where a visibility switch will occur. While this paper handles the problem by
improving differentiation through discontinuities, our project could explore
another direction by explicitly learning and locating the event boundaries
themselves.

---

## 5. Differentiable Shadow Mapping for Efficient Inverse Graphics

**Worchel and Alexa (2023), CVPR**

This paper introduces differentiable shadow mapping for inverse graphics. It
makes shadow information usable during optimization and shows that shadows can
provide useful constraints for estimating unknown scene or illumination
parameters.

This is especially relevant to the inversion part of our project. Observed
shadows could be used to infer possible light parameters, but the inferred
candidates can also be independently checked using a physical ray tracer. If a
candidate is rejected, the disagreement may reveal an error in the learned
visibility-event geometry and provide useful information for deciding where
additional simulation queries should be added.

---

## Overall Idea

Based on these papers, I think a promising direction is to connect event
learning with inversion and physical validation. Instead of treating inversion
only as a final application, rejected inversion candidates could help identify
important errors in the learned event boundaries and guide additional
ray-tracing queries. The overall idea is to test whether this
inversion-guided refinement can improve event learning and transfer to unseen
structures under a limited query budget.
