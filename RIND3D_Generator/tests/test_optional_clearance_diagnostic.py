"""An optional diagnostic timeout must never become a physical classification."""
import sys,time
from pathlib import Path
from rind3d.geometry import Oracle
from rind3d.benchmark import fixture_box
from rind3d.windows import Window

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_origin_feature_audit import optional_query,BoundedClearance
from origin_audit_meter import AuditMeter
from origin_distribution_features import measure_center_clearance


def test_completed_optional_query_preserves_payload():
    result,status,cpu,wall=optional_query(lambda:{'certified':True,'value':[1,0]},1)
    assert status=='COMPLETE' and result=={'certified':True,'value':[1,0]}
    assert cpu>=0 and wall>0


def test_optional_timeout_returns_no_partial_classification_and_reaps_child():
    def slow():time.sleep(.3);return {'near':False}
    result,status,cpu,wall=optional_query(slow,.02)
    assert result is None and status=='DIAGNOSTIC_TIMEOUT'
    assert cpu>=0 and .02<=wall<1


def test_optional_error_is_not_a_far_or_zero_value():
    def unresolved():raise RuntimeError('unresolved exact distance')
    result,status,_,_=optional_query(unresolved,1)
    assert result is None and status.startswith('DIAGNOSTIC_ERROR:RuntimeError:')


def test_completed_bounded_descriptor_matches_original_canonical_query():
    meter=AuditMeter(Oracle(fixture_box()));window=Window((400,400,400),64)
    expected=measure_center_clearance(window,meter,32/1024)
    bounded=BoundedClearance(1.)
    with meter.timed('diagnostic_clearance'):actual=bounded.measure(window,meter,32/1024)
    assert actual['clearance_measurement_status']=='COMPLETE'
    assert all(actual[k]==v for k,v in expected.items())
    assert actual['center_near_solid_r32'] is not None
    assert bounded.child_cpu>=0 and meter.report()['calls']['diagnostic_clearance']['clearance_calls']==1
    meter.close()
