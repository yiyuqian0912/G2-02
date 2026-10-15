"""Canonical Phase I energy field; inputs and saved candidates use world coordinates."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FourierXY(nn.Module):
    def __init__(self, frequencies=6):
        super().__init__()
        if frequencies < 0:
            raise ValueError('frequencies must be nonnegative')
        self.frequencies = frequencies

    def forward(self, xy):
        out = [xy]
        for k in range(self.frequencies):
            w = 2.0 ** k * math.pi
            out.extend((torch.sin(w * xy), torch.cos(w * xy)))
        return torch.cat(out, dim=-1)


class ObservationEncoder(nn.Sequential):
    def __init__(self, hidden=128, channels=2):
        super().__init__(nn.Conv2d(channels, 32, 3, padding=1), nn.GELU(),
                         nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                         nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.GELU())


class ResidualConvBlock(nn.Module):
    """Residual image block used only by the opt-in large student."""
    def __init__(self, channels):
        super().__init__()
        groups=math.gcd(32,channels)
        self.body=nn.Sequential(
            nn.Conv2d(channels,channels,3,padding=1,bias=False),
            nn.GroupNorm(groups,channels),nn.GELU(),
            nn.Conv2d(channels,channels,3,padding=1,bias=False),
            nn.GroupNorm(groups,channels))
        self.activation=nn.GELU()

    def forward(self,x):
        return self.activation(x+self.body(x))


class LargeObservationEncoder(nn.Sequential):
    """Deeper downsampling encoder for high-density candidate training."""
    def __init__(self,hidden=256,channels=2):
        groups64=math.gcd(32,64);groups128=math.gcd(32,128);groupsh=math.gcd(32,hidden)
        super().__init__(
            nn.Conv2d(channels,64,5,padding=2,bias=False),nn.GroupNorm(groups64,64),nn.GELU(),
            ResidualConvBlock(64),
            nn.Conv2d(64,128,3,stride=2,padding=1,bias=False),nn.GroupNorm(groups128,128),nn.GELU(),
            ResidualConvBlock(128),
            nn.Conv2d(128,hidden,3,stride=2,padding=1,bias=False),nn.GroupNorm(groupsh,hidden),nn.GELU(),
            ResidualConvBlock(hidden),
            nn.Conv2d(hidden,hidden,3,stride=2,padding=1,bias=False),nn.GroupNorm(groupsh,hidden),nn.GELU(),
            ResidualConvBlock(hidden),ResidualConvBlock(hidden))


class CandidateAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.keys = nn.Linear(hidden, hidden)
        self.values = nn.Linear(hidden, hidden)

    def forward(self, query, tokens):
        keys = self.keys(tokens)
        values = self.values(tokens)
        return F.scaled_dot_product_attention(
            query[:, None], keys[:, None], values[:, None]).squeeze(1)


class SceneGeometryEncoder(nn.Module):
    """Encode the exact obstacle primitive set without source information."""
    def __init__(self, hidden=128, parameter_width=32, type_count=4,
                 attention_heads=4, layers=2, dropout=0.0):
        super().__init__()
        if (parameter_width < 1 or type_count < 1 or layers < 0 or
                not math.isfinite(dropout) or dropout < 0 or dropout >= 1):
            raise ValueError('invalid scene geometry encoder dimensions')
        if attention_heads < 1 or hidden % attention_heads:
            raise ValueError('geometry attention heads must divide hidden')
        type_width=max(8,hidden//8)
        self.parameter_width=int(parameter_width)
        self.type_count=int(type_count)
        self.type_embedding=nn.Embedding(type_count,type_width)
        self.input=nn.Sequential(nn.Linear(parameter_width+type_width,hidden),nn.GELU(),
                                 nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
        if layers:
            layer=nn.TransformerEncoderLayer(
                d_model=hidden,nhead=attention_heads,dim_feedforward=4*hidden,
                dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
            self.context=nn.TransformerEncoder(layer,num_layers=layers,
                                               enable_nested_tensor=False)
        else:
            self.context=nn.Identity()
        self.norm=nn.LayerNorm(hidden)

    def forward(self, obstacle_types, obstacle_params, obstacle_mask,
                coordinate_scale):
        if (obstacle_types.ndim!=2 or obstacle_params.ndim!=3 or
                obstacle_params.shape[:2]!=obstacle_types.shape or
                obstacle_params.shape[-1]!=self.parameter_width or
                obstacle_mask.shape!=obstacle_types.shape or
                obstacle_mask.dtype!=torch.bool):
            raise ValueError('scene geometry must be types [B,M], params [B,M,P], mask [B,M]')
        if obstacle_types.dtype!=torch.long:
            raise ValueError('scene obstacle types must use torch.long')
        if (not obstacle_mask.any(-1).all() or
                not torch.isfinite(obstacle_params).all() or
                torch.any(obstacle_types[obstacle_mask]<0) or
                torch.any(obstacle_types[obstacle_mask]>=self.type_count)):
            raise ValueError('invalid scene obstacle tokens')
        # RIND primitive parameters are type-specific. Coordinate-like values
        # are scaled to the world; rectangle/ellipse angles and polygon vertex
        # counts receive their own natural normalization.
        normalized=obstacle_params/float(coordinate_scale)
        rect_or_ellipse=(obstacle_types==0)|(obstacle_types==1)
        normalized=normalized.clone()
        normalized[...,4]=torch.where(
            rect_or_ellipse,obstacle_params[...,4]/math.pi,normalized[...,4])
        polygon=obstacle_types==3
        normalized[...,0]=torch.where(
            polygon,obstacle_params[...,0]/12.0,normalized[...,0])
        safe_types=obstacle_types.clamp(0,self.type_count-1)
        tokens=self.input(torch.cat((normalized,self.type_embedding(safe_types)),-1))
        tokens=tokens.masked_fill(~obstacle_mask[...,None],0)
        if not isinstance(self.context,nn.Identity):
            tokens=self.context(tokens,src_key_padding_mask=~obstacle_mask)
        return self.norm(tokens).masked_fill(~obstacle_mask[...,None],0)


def student_distribution(energy, valid_mask=None):
    if energy.ndim != 2 or energy.shape[1] == 0:
        raise ValueError('energy must have shape [B,K], K > 0')
    mask = torch.ones_like(energy, dtype=torch.bool) if valid_mask is None else valid_mask
    if mask.shape != energy.shape or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('mask must be bool [B,K] with at least one valid candidate per row')
    if not torch.isfinite(energy[mask]).all():
        raise ValueError('valid energies must be finite')
    log_prob = (-energy).masked_fill(~mask, -torch.inf).log_softmax(-1)
    return log_prob, log_prob.exp()


def fuse_view_log_probabilities(log_probability, view_mask=None, weights=None,
                                log_prior=None):
    """Fuse independent views of one contribution by adding log evidence.

    ``log_probability`` is ``[B,V,N]``.  With a uniform prior this is the
    normalized product of the selected per-view posteriors.  A non-uniform
    prior may be supplied as ``[N]`` or ``[B,N]``; it is removed for every
    additional view so the prior is counted exactly once.  Association of
    residual layers across views is intentionally an upstream responsibility.
    """
    if log_probability.ndim != 3 or log_probability.shape[-1] == 0:
        raise ValueError('log_probability must have shape [B,V,N], N > 0')
    batch,views,_=log_probability.shape
    if view_mask is None:
        view_mask=torch.ones(batch,views,dtype=torch.bool,device=log_probability.device)
    if (view_mask.shape!=(batch,views) or view_mask.dtype!=torch.bool or
            not view_mask.any(-1).all()):
        raise ValueError('view_mask must be bool [B,V] with at least one view per item')
    if weights is None:
        weights=torch.ones(batch,views,dtype=log_probability.dtype,
                           device=log_probability.device)
    if (weights.shape!=(batch,views) or not torch.isfinite(weights).all() or
            torch.any(weights<0)):
        raise ValueError('weights must be finite nonnegative [B,V]')
    active_weights=weights*view_mask.to(weights.dtype)
    if not (active_weights.sum(-1)>0).all():
        raise ValueError('selected view weights must have positive total weight')
    selected=torch.where(view_mask[...,None],log_probability,
                         torch.zeros((),dtype=log_probability.dtype,
                                     device=log_probability.device))
    if not torch.isfinite(selected).all():
        raise ValueError('selected view log probabilities must be finite')
    evidence=(selected*active_weights[...,None]).sum(1)
    total_weight=active_weights.sum(-1,keepdim=True)
    if log_prior is not None:
        if log_prior.ndim==1:
            log_prior=log_prior[None].expand(batch,-1)
        if log_prior.shape!=evidence.shape or not torch.isfinite(log_prior).all():
            raise ValueError('log_prior must be finite [N] or [B,N]')
        evidence=evidence-(total_weight-1)*log_prior
    fused_log=evidence.log_softmax(-1)
    return fused_log,fused_log.exp()


class SourceEnergyField(nn.Module):
    def __init__(self, hidden=128, fourier_frequencies=6, coordinate_scale=1024.0,
                 use_obstacle=True, pixel_coordinates=False, normalization_support="geometry",
                 architecture="baseline", attention_heads=1,
                 use_scene_geometry=False, geometry_parameter_width=32,
                 geometry_type_count=4, geometry_attention_heads=4,
                 geometry_layers=2, geometry_dropout=0.0,
                 geometry_fusion='concat',geometry_cross_attention_heads=None):
        super().__init__()
        if hidden < 2:
            raise ValueError('hidden must be at least 2')
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError('coordinate_scale must be positive')
        if normalization_support not in ('world', 'geometry'):
            raise ValueError('normalization_support must be world or geometry')
        if architecture not in ('baseline','large'):
            raise ValueError('architecture must be baseline or large')
        if type(attention_heads) is not int or attention_heads<1 or hidden%attention_heads:
            raise ValueError('attention_heads must be a positive divisor of hidden')
        if architecture=='large' and hidden<64:
            raise ValueError('large architecture requires hidden >= 64')
        if type(use_scene_geometry) is not bool:
            raise ValueError('use_scene_geometry must be boolean')
        if (type(geometry_attention_heads) is not int or geometry_attention_heads<1 or
                hidden%geometry_attention_heads):
            raise ValueError('geometry_attention_heads must divide hidden')
        if type(geometry_layers) is not int or geometry_layers<0:
            raise ValueError('geometry_layers must be a nonnegative integer')
        if (not math.isfinite(geometry_dropout) or geometry_dropout<0 or
                geometry_dropout>=1):
            raise ValueError('geometry_dropout must be in [0,1)')
        if geometry_fusion not in ('concat','residual'):
            raise ValueError('geometry_fusion must be concat or residual')
        if not use_scene_geometry and geometry_fusion!='concat':
            raise ValueError('geometry_fusion requires use_scene_geometry')
        geometry_cross_attention_heads=(attention_heads if geometry_cross_attention_heads is None
                                        else geometry_cross_attention_heads)
        if (type(geometry_cross_attention_heads) is not int or
                geometry_cross_attention_heads<1 or hidden%geometry_cross_attention_heads):
            raise ValueError('geometry cross-attention heads must divide hidden')
        self.use_obstacle = bool(use_obstacle)
        self.use_scene_geometry=use_scene_geometry
        self.pixel_coordinates = bool(pixel_coordinates)
        self.normalization_support = normalization_support
        self.coordinate_scale = float(coordinate_scale)
        self.architecture=architecture
        self.attention_heads=attention_heads
        self.geometry_fusion=geometry_fusion
        self.geometry_cross_attention_heads=geometry_cross_attention_heads
        self.interface_version = 'phase1-v1'
        if architecture=='large':
            self.window_encoder=nn.Sequential(nn.Linear(3,hidden),nn.GELU(),nn.LayerNorm(hidden),
                                              nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
            self.image_encoder=LargeObservationEncoder(hidden,4 if self.pixel_coordinates else 2)
        else:
            self.window_encoder = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.image_encoder = ObservationEncoder(hidden, 4 if self.pixel_coordinates else 2)
        self.xy = FourierXY(fourier_frequencies)
        coordinate_features=2*(1+2*fourier_frequencies)
        if architecture=='large':
            self.source_query=nn.Sequential(nn.Linear(coordinate_features,hidden),nn.GELU(),
                                            nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),
                                            nn.Linear(hidden,hidden))
        else:
            self.source_query = nn.Sequential(nn.Linear(coordinate_features, hidden),
                                              nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = CandidateAttention(hidden)
        if use_scene_geometry:
            self.geometry_encoder=SceneGeometryEncoder(
                hidden,geometry_parameter_width,geometry_type_count,
                geometry_attention_heads,geometry_layers,geometry_dropout)
            self.geometry_attention=CandidateAttention(hidden)
        head_input=(3 if use_scene_geometry and geometry_fusion=='concat' else 2)*hidden
        if architecture=='large':
            self.energy_head=nn.Sequential(nn.Linear(head_input,2*hidden),nn.GELU(),nn.LayerNorm(2*hidden),
                                           nn.Linear(2*hidden,hidden),nn.GELU(),
                                           nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        else:
            self.energy_head = nn.Sequential(nn.Linear(head_input, hidden), nn.GELU(),
                                             nn.Linear(hidden, hidden // 2), nn.GELU(),
                                             nn.Linear(hidden // 2, 1))
        if use_scene_geometry and geometry_fusion=='residual':
            if architecture=='large':
                self.geometry_correction_head=nn.Sequential(
                    nn.Linear(3*hidden,hidden),nn.GELU(),nn.LayerNorm(hidden),
                    nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
            else:
                self.geometry_correction_head=nn.Sequential(
                    nn.Linear(3*hidden,hidden),nn.GELU(),
                    nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
            # The geometry-aware model initially reproduces the blind model
            # exactly.  Training must earn every geometry-dependent change.
            nn.init.zeros_(self.geometry_correction_head[-1].weight)
            nn.init.zeros_(self.geometry_correction_head[-1].bias)

    def encode_observation(self, response, obstacle, window, geometry_types=None,
                           geometry_params=None, geometry_mask=None):
        """Encode once for any number of candidates; legacy weights remain loadable."""
        if response.ndim != 3 or obstacle.shape != response.shape or obstacle.dtype != torch.bool:
            raise ValueError('response/obstacle must be [B,size,size], obstacle boolean')
        if window.shape != (len(response),3) or response.shape[1] != response.shape[2] or not (window[:,2] == response.shape[1]).all():
            raise ValueError('window must match the native response size')
        if not torch.isfinite(response).all() or not torch.isfinite(window).all():
            raise ValueError('inputs must be finite')
        mask = obstacle.to(response.dtype) if self.use_obstacle else torch.zeros_like(response)
        channels = [response,mask]
        if self.pixel_coordinates:
            size=response.shape[-1]
            axis=(torch.arange(size,device=response.device,dtype=response.dtype)+.5)/size*2-1
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            channels += [xx.expand_as(response), yy.expand_as(response)]
        tokens=self.image_encoder(torch.stack(channels,dim=1)).flatten(2).transpose(1,2)
        encoded=(self.attention.keys(tokens),self.attention.values(tokens),
                 self.window_encoder(window.to(response.dtype)/self.coordinate_scale)[:,None])
        supplied=(geometry_types is not None or geometry_params is not None or geometry_mask is not None)
        if self.use_scene_geometry:
            if not supplied or geometry_types is None or geometry_params is None or geometry_mask is None:
                raise ValueError('full scene obstacle geometry is required by this model')
            geometry_tokens=self.geometry_encoder(
                geometry_types,geometry_params.to(response.dtype),geometry_mask,self.coordinate_scale)
            encoded += (self.geometry_attention.keys(geometry_tokens),
                        self.geometry_attention.values(geometry_tokens),geometry_mask)
        elif supplied:
            raise ValueError('scene geometry was supplied to a model that does not use it')
        return encoded

    def score_encoded(self, encoded, candidate_xy):
        if len(encoded) not in (3,6):
            raise ValueError('invalid encoded observation tuple')
        keys,values,window_feature=encoded[:3]
        if candidate_xy.ndim != 3 or candidate_xy.shape[0] != len(keys) or candidate_xy.shape[-1] != 2 or candidate_xy.shape[1] == 0:
            raise ValueError('candidate_xy must be [B,N,2], N>0')
        if not torch.isfinite(candidate_xy).all():
            raise ValueError('candidate coordinates must be finite')
        query=self.source_query(self.xy(candidate_xy/self.coordinate_scale))+window_feature
        batch,candidates,hidden=query.shape;tokens=keys.shape[1];head=hidden//self.attention_heads
        q=query.reshape(batch,candidates,self.attention_heads,head).transpose(1,2)
        k=keys.reshape(batch,tokens,self.attention_heads,head).transpose(1,2)
        v=values.reshape(batch,tokens,self.attention_heads,head).transpose(1,2)
        # PyTorch selects a fused scaled-dot-product attention kernel on CUDA.
        # Besides avoiding the explicit [B,H,candidates,tokens] score tensor,
        # this uses the backend's numerically stable softmax implementation.
        context=F.scaled_dot_product_attention(q,k,v)
        context=context.transpose(1,2).reshape(batch,candidates,hidden)
        features=[query,context]
        geometry_context=None
        if len(encoded)==6:
            geometry_keys,geometry_values,geometry_mask=encoded[3:]
            geometry_tokens=geometry_keys.shape[1]
            geometry_head=hidden//self.geometry_cross_attention_heads
            geometry_q=query.reshape(
                batch,candidates,self.geometry_cross_attention_heads,geometry_head).transpose(1,2)
            gk=geometry_keys.reshape(
                batch,geometry_tokens,self.geometry_cross_attention_heads,geometry_head).transpose(1,2)
            gv=geometry_values.reshape(
                batch,geometry_tokens,self.geometry_cross_attention_heads,geometry_head).transpose(1,2)
            attention_bias=torch.zeros((batch,1,1,geometry_tokens),dtype=query.dtype,device=query.device)
            attention_bias.masked_fill_(~geometry_mask[:,None,None,:],-torch.inf)
            geometry_context=F.scaled_dot_product_attention(
                geometry_q,gk,gv,attn_mask=attention_bias)
            geometry_context=geometry_context.transpose(1,2).reshape(batch,candidates,hidden)
            if self.geometry_fusion=='concat':
                features.append(geometry_context)
        base_energy=self.energy_head(torch.cat(features,-1)).squeeze(-1)
        if geometry_context is None or self.geometry_fusion=='concat':
            return base_energy
        correction=self.geometry_correction_head(
            torch.cat((query,context,geometry_context),-1)).squeeze(-1)
        return base_energy+correction

    def forward(self, response, obstacle, window, candidate_xy, geometry_types=None,
                geometry_params=None, geometry_mask=None):
        return self.score_encoded(self.encode_observation(
            response,obstacle,window,geometry_types,geometry_params,geometry_mask),candidate_xy)

    def predict(self, response, obstacle, window, candidate_xy, valid=None,
                geometry_types=None, geometry_params=None, geometry_mask=None):
        energy = self(response, obstacle, window, candidate_xy,
                      geometry_types,geometry_params,geometry_mask)
        support = valid if self.normalization_support == 'geometry' else None
        log_prob, prob = student_distribution(energy, support)
        return {'energy': energy, 'student_prob': prob, 'student_log_prob': log_prob}
