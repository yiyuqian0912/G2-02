"""Bounded, soft coverage/overlap optimization; no validity or spacing gates.

Objective per side: sum_cells log(1+n_cell) - weight * sum_pairs IoU.
Lazy greedy marginals are monotone. Bounded swaps only improve this objective;
they cannot alter side quotas or admit a candidate outside the exact-valid pool.
"""
from collections import Counter,defaultdict
from itertools import product
import heapq,math,time
import numpy as np


class SpatialSelection:
    def __init__(self,origins,side,cell_size,weight):
        self.origins=np.asarray(origins,dtype=np.int32).reshape(-1,3);self.centers=self.origins+side/2
        self.side=side;self.cell_size=cell_size;self.weight=weight
        self.cells=np.floor(self.centers/cell_size).astype(int);self.hash_cells=np.floor(self.centers/side).astype(int)
        self.coverage=Counter();self.hash=defaultdict(set);self.selected=set();self.evaluations=0
    def overlaps(self,index):
        cell=self.hash_cells[index];neighbors=[]
        for delta in product((-1,0,1),repeat=3):neighbors.extend(self.hash.get(tuple(cell+delta),()))
        if not neighbors:return 0.
        neighbors=[j for j in neighbors if j!=index]
        if not neighbors:return 0.
        d=np.abs(self.origins[neighbors]-self.origins[index]);intersection=np.maximum(0,self.side-d).prod(axis=1,dtype=np.int64)
        return float((intersection/(2*self.side**3-intersection)).sum())
    def gain(self,index):
        self.evaluations+=1;n=self.coverage[tuple(self.cells[index])]
        return math.log1p(1/(1+n))-self.weight*self.overlaps(index)
    def add(self,index):
        self.selected.add(index);self.coverage[tuple(self.cells[index])]+=1;self.hash[tuple(self.hash_cells[index])].add(index)
    def remove(self,index):
        self.selected.remove(index);self.coverage[tuple(self.cells[index])]-=1;self.hash[tuple(self.hash_cells[index])].remove(index)


def soft_select(origins,side,target,settings,seed):
    """No exponential assignment, bucket cap, or minimum pairwise distance."""
    start=time.perf_counter();target=min(target,len(origins));state=SpatialSelection(origins,side,settings['coverage_cell_side'],settings['overlap_weight'])
    if target==len(origins):
        chosen=list(range(target));cells={tuple(c) for c in state.cells}
        return chosen,dict.fromkeys(chosen,None),{'mode':'soft','optimization_applied':False,'reason':'all_pool_candidates_required',
            'score_evaluations':0,'swap_attempts':0,'swaps':[],'occupied_center_cells':len(cells),
            'hard_bucket_cap':None,'hard_center_spacing':None,'wall_seconds':time.perf_counter()-start}
    rng=np.random.default_rng(seed);ties=rng.random(len(origins));heap=[(-math.log(2),float(ties[i]),i) for i in range(len(origins))];heapq.heapify(heap)
    scores={};order=[]
    capped=0
    while len(order)<target:
        evaluated={}
        while True:
            while heap and heap[0][2] in state.selected:heapq.heappop(heap)
            _,tie,index=heapq.heappop(heap)
            if index not in evaluated:evaluated[index]=state.gain(index)
            score=evaluated[index]
            while heap and heap[0][2] in state.selected:heapq.heappop(heap)
            if not heap or (-score,tie,index)<=heap[0]:break
            heapq.heappush(heap,(-score,tie,index))
            if len(evaluated)>=settings['greedy_evaluations_per_selected']:
                index=min(evaluated,key=lambda i:(-evaluated[i],ties[i],i));score=evaluated[index];capped+=1;break
        state.add(index);order.append(index);scores[index]=score
    # Coverage-first alternatives are tried against redundant/dense selected cells.
    # The exact objective delta uses spatial neighbors with the old point removed.
    attempts=0;swaps=[];unselected=[i for i in rng.permutation(len(origins)).tolist() if i not in state.selected]
    limit=min(len(unselected),settings['swap_attempts_per_selected']*target)
    if target:
        for candidate in unselected[:limit]:
            attempts+=1
            # A bounded rotating shortlist avoids a global combinatorial solve.
            chosen=[order[(attempts*17+j*max(1,target//16))%target] for j in range(min(16,target))]
            old=max(chosen,key=lambda i:(state.coverage[tuple(state.cells[i])],-scores.get(i,0),i))
            cell=tuple(state.cells[old]);n=state.coverage[cell]
            removal=-math.log1p(1/n)+state.weight*state.overlaps(old)
            state.remove(old);addition=state.gain(candidate)
            if removal+addition>0:
                state.add(candidate);order[order.index(old)]=candidate;scores.pop(old,None);scores[candidate]=addition
                swaps.append({'removed_pool_index':old,'added_pool_index':candidate,'objective_gain':removal+addition})
            else:state.add(old)
    return order,scores,{'mode':'soft','objective':'sum log(1+cell occupancy) - weight * pairwise IoU',
        'coverage_cell_side':state.cell_size,'overlap_weight':state.weight,'score_evaluations':state.evaluations,
        'swap_attempts':attempts,'swaps':swaps,'occupied_center_cells':sum(n>0 for n in state.coverage.values()),
        'bounded_approximate_greedy':True,'greedy_evaluations_per_selected':settings['greedy_evaluations_per_selected'],'capped_selection_steps':capped,
        'hard_bucket_cap':None,'hard_center_spacing':None,'wall_seconds':time.perf_counter()-start}
