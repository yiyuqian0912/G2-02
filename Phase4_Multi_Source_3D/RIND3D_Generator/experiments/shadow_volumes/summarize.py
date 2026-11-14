"""Measured summaries, explicit verification scope, and conditional capacity estimates."""
import csv,json,statistics
from pathlib import Path
from rind3d.pack import atomic_json,HEADER,RECORD,RAW
from .benchmark import ROOT,compare
from .cases import NAMES


def read(path):return json.loads(Path(path).read_text())


def main():
    small=[read(ROOT/'small'/name/'result.json') for name in NAMES]
    full=[]
    for p in sorted((ROOT/'full').glob('*/*/result.json')):
        v=read(p);v.update(family=p.parent.parent.name,result_path=str(p));full.append(v)
    pairs=[]
    for name in ('box','corridor'):
        for method in ('B','B_native'):
            a=ROOT/'full'/name/'C-source0';b=ROOT/'full'/name/(method+'-source0')
            if read(a/'result.json')['complete'] and read(b/'result.json')['complete']:
                compare(a,b);pairs.append({'family':name,'methods':['C',method],'resolution':1024,'all_voxels_equal':True})
    strat=[read(p) for p in sorted((ROOT/'stratified').glob('*/result.json'))]
    reuse_archive=[read(p) for folder in ('reuse-1024','reuse-128') for p in sorted((ROOT/folder).glob('*/result.json')) if '-timing' not in p.parent.name]
    reuse=[read(ROOT/f'reuse-{n}'/(name+'-timing')/'result.json') for name,n in (('rotated_box',1024),('poly16',128))]
    geom_only=read(ROOT/'reuse-1024/rotated_box/geometry-only-control-timing.json')
    lite=read(ROOT/'lite-compatibility/result.json');diversity=read(ROOT/'diversity-statistics.json')
    rows=[]
    for v in full:
        g=v.get('generation',{});setup=v['total_wall_including_setup']-v['wall_seconds'];r={k:v.get(k) for k in ('family','method','status','complete','seconds_budget','geometry_init_seconds','total_wall_including_setup','total_cpu_including_setup','peak_rss_bytes','committed_brick_pairs')}
        r.update(first_brick_including_setup=setup+v['first_brick_seconds'] if v['first_brick_seconds'] is not None else None,
            field_seconds=v['wall_seconds'],mean_worker_job_seconds=g.get('render_worker_seconds',0)/4096 if v['complete'] else None,
            certificate_coverage=g.get('certificate_covered_voxels',0)/1024**3 if v['complete'] else None,
            file_bytes=sum(g.get('field_bytes',{}).values()) if v['complete'] else None,
            canonical_fallback_rays=None if v['method']=='A' else v.get('work_this_field',v['acceleration']['work']).get('canonical_fallback_rays',0),
            shadow_planes=sum(v['acceleration']['shadow_plane_counts']))
        rows.append(r)
    with (ROOT/'full-fields.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    completed=[v for v in full if v['method']=='C' and v['complete']]
    cpu=[v['total_cpu_including_setup'] for v in completed];wall=[v['total_wall_including_setup'] for v in completed]
    demo=ROOT/'lite-compatibility';metadata_bytes=sum(p.stat().st_size for p in demo.rglob('*') if p.is_file() and p.name not in ('response.rindb','solid.rindb','information.rindi','geometry.npz','result.json','progress.json'))
    disk=[sum(v['generation']['field_bytes'].values())+(Path(v['result_path']).parent/'information.rindi').stat().st_size+(ROOT/'cases'/v['family']/'geometry.npz').stat().st_size+metadata_bytes for v in completed]
    projections={}
    for count in (5000,10000):
        projections[count]={
            'field_only_cpu_hours_min_median_max':[count*f(cpu)/3600 for f in (min,statistics.median,max)],
            'field_only_serial_wall_hours_min_median_max':[count*f(wall)/3600 for f in (min,statistics.median,max)],
            'illustrative_truth_only_Lite_cpu_hours_min_median_max':[count*(f(cpu)+lite['process_seconds'])/3600 for f in (min,statistics.median,max)],
            'compressed_bytes_min_median_max':[int(count*f(disk)) for f in (min,statistics.median,max)],
            'pack_raw_upper_bound_two_fields_bytes':count*2*(HEADER+4096*RECORD.size+4096*RAW),
            'scope':'Measured completed families only; one source/geometry. Lite overhead measured on one box with one verified source/window; extra seven candidates, scene acceptance/rejections, population distribution and production-scale I/O not measured. Index rebuilding is conservatively charged twice.',
        }
    atomic_json(ROOT/'summary.json',{'small':small,'full':full,'full_table':rows,'full_original_control_bit_comparisons':pairs,'stratified':strat,'reuse':reuse,'reuse_archived_with_comparison_overhead':reuse_archive,'geometry_only_reuse':geom_only,'lite':lite,'diversity':diversity,'projections':projections,
        'small_independent_comparisons':36,'small_logical_voxels_compared':12*(32**3+64**3+128**3),'canonical_random_ray_checks':sum(v['canonical_checks']['queries'] for v in small),
        'canonical_exhaustive_scope':'Completed original-filter B / B_native controls for axis-aligned boxes and corridor. Rotated/general-polyhedron CGAL-only full fields did not complete; independent clipping comparisons do NOT close that coverage gap.',
        'no_mass_production':True,'production_ready':False})
    print('Summarized',len(full),'full-field attempts,',len(pairs),'original-control all-bit comparisons')


if __name__=='__main__':main()
