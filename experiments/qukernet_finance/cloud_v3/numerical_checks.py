"""Independent local analytic checks: no financial data, authentication or cloud.

From experiments/qukernet_finance, run explicitly:
    python -m cloud_v3.numerical_checks
The default prints JSON without writing a file. An optional --output accepts a
new private JSON destination. Teaching notebooks never invoke this module.
Both independent state evaluators, fixed cases, tolerances and mathematical
checks reproduce the original local numerical validation without alterations.
"""
from pathlib import Path
import hashlib
import json
import math
import argparse
from datetime import datetime, timezone

import numpy as np

from cloud_v2 import circuits as qc
from cloud_v2 import search
from cloud_v3 import study


def rotation(axis, theta):
    ident=np.eye(2,dtype=complex)
    pauli={'RX':np.array([[0,1],[1,0]],complex),
           'RY':np.array([[0,-1j],[1j,0]],complex),
           'RZ':np.diag([1,-1]).astype(complex)}[axis]
    return np.cos(theta/2)*ident-1j*np.sin(theta/2)*pauli


def apply_internal(gates, qubits):
    """Tensor-axis evolution; internal wire zero is most significant."""
    state=np.zeros(2**qubits,complex);state[0]=1
    for gate in gates:
        if gate['gate'] in ('RX','RY','RZ'):
            wire=gate['wires'][0]
            shaped=state.reshape([2]*qubits)
            moved=np.moveaxis(shaped,wire,0).reshape(2,-1)
            updated=rotation(gate['gate'],gate['angle'])@moved
            state=np.moveaxis(updated.reshape([2]+[2]*(qubits-1)),0,wire).reshape(-1)
        elif gate['gate']=='CNOT':
            control,target=gate['wires']
            permutation=np.arange(2**qubits)
            active=(permutation & (1 << (qubits-1-control)))!=0
            permutation[active]^=1 << (qubits-1-target)
            state=state[permutation]
        else:
            raise ValueError('Unsupported internal gate')
    return state


def apply_qcis(text, qubits):
    """Independent bit-index QCIS parser; CNOT arrives as H-CZ-H."""
    state=np.zeros(2**qubits,complex);state[0]=1
    measurements=[]
    for line in text.splitlines():
        parts=line.split();name=parts[0]
        if name=='M':
            measurements.append(int(parts[1][1:]));continue
        if name=='CZ':
            a,b=[int(p[1:])-1 for p in parts[1:]]
            for basis in range(2**qubits):
                if basis & (1<<(qubits-1-a)) and basis & (1<<(qubits-1-b)):
                    state[basis]*=-1
            continue
        wire=int(parts[1][1:])-1;bit=1<<(qubits-1-wire)
        if name=='H':
            matrix=np.asarray([[1,1],[1,-1]],complex)/np.sqrt(2)
        else:
            angle=float(parts[2]);c,s=np.cos(angle/2),np.sin(angle/2)
            matrix={'RX':np.asarray([[c,-1j*s],[-1j*s,c]]),
                    'RY':np.asarray([[c,-s],[s,c]],complex),
                    'RZ':np.diag([np.exp(-1j*angle/2),np.exp(1j*angle/2)])}[name]
        for basis in range(2**qubits):
            if not basis & bit:
                pair=np.asarray([state[basis],state[basis|bit]])
                state[basis],state[basis|bit]=matrix@pair
    if measurements!=list(range(1,qubits+1)):
        raise ValueError('Measurement labels differ')
    return state


def kernel_matrix(x, layout, mask, params):
    states=[apply_internal(qc.feature_gates(row,layout,mask,params),layout['qubits']) for row in x]
    return np.abs(np.asarray(states).conj()@np.asarray(states).T)**2


def exact_pair_shift(xa,xb,layout,mask,params):
    gates=qc.overlap_gates(xa,xb,layout,mask,params)
    out=np.zeros(len(mask))
    occurrence_counts=[]
    for p in range(len(mask)):
        occurrences=qc.shift_occurrences(gates,p)
        ends=[gates[i]['end'] for i,_ in occurrences]
        assert set(ends)=={'left','right'}
        occurrence_counts.append(len(occurrences))
        for index,coefficient in occurrences:
            values={}
            for sign in (-1,1):
                text=qc.qcis(gates,layout['qubits'],(index,sign*math.pi/2))
                values[sign]=abs(apply_qcis(text,layout['qubits'])[0])**2
            out[p]+=.5*coefficient*(values[1]-values[-1])
    return out,occurrence_counts


def run_checks():
    if not __debug__:
        raise RuntimeError("Run without Python -O so numerical assertions remain enabled")
    rng=np.random.default_rng(20261115)
    cases=[]
    for qubits,depth in ((3,2),(8,1),(8,3)):
        layout={'qubits':qubits,'features':qubits,'depth':depth,
                'blocks':[{'axes':[('RX','RY','RZ')[i%3] for i in range(qubits)],
                           'entanglers':[1]*(qubits-1)}]}
        x=rng.uniform(-1.3,1.4,(4,qubits))
        # Include distinct physical positions from a repeat for the deep case.
        mask=[0,(depth-1)*qubits+min(1,qubits-1)]
        params=qc.initial_parameters(x,layout,mask)+np.asarray([.17,-.23])
        probability_errors=[];state_errors=[];fd_errors=[]
        derivatives=np.zeros((2,4,4));counts=[]
        epsilon=1e-6
        for i in range(4):
            for j in range(i+1,4):
                left=apply_internal(qc.feature_gates(x[i],layout,mask,params),qubits)
                right=apply_internal(qc.feature_gates(x[j],layout,mask,params),qubits)
                p_overlap=abs(np.vdot(right,left))**2
                gates=qc.overlap_gates(x[i],x[j],layout,mask,params)
                internal=apply_internal(gates,qubits)
                independent=apply_qcis(qc.qcis(gates,qubits),qubits)
                state_errors.append(float(np.max(np.abs(internal-independent))))
                probability_errors.append(abs(p_overlap-abs(independent[0])**2))
                shift,occurrences=exact_pair_shift(x[i],x[j],layout,mask,params)
                counts.append(occurrences)
                for p in range(2):
                    delta=np.zeros(2);delta[p]=epsilon
                    kp=kernel_matrix(x[[i,j]],layout,mask,params+delta)[0,1]
                    km=kernel_matrix(x[[i,j]],layout,mask,params-delta)[0,1]
                    fd_errors.append(abs(shift[p]-(kp-km)/(2*epsilon)))
                    derivatives[p,i,j]=derivatives[p,j,i]=shift[p]
        k=kernel_matrix(x,layout,mask,params);y=np.asarray([0,1,0,1])
        value,gradient=search.alignment_and_gradient(k,y,'classification',derivatives)
        signed=2*y-1;target=np.outer(signed,signed)
        manual=float((k*target).sum()/(np.linalg.norm(k)*np.linalg.norm(target)))
        objective_fd=[]
        for p in range(2):
            delta=np.zeros(2);delta[p]=epsilon
            values=[]
            for theta in (params-delta,params+delta):
                kk=kernel_matrix(x,layout,mask,theta)
                values.append(float((kk*target).sum()/(np.linalg.norm(kk)*np.linalg.norm(target))))
            objective_fd.append((values[1]-values[0])/(2*epsilon))
        objective_error=float(np.max(np.abs(gradient-objective_fd)))
        serialized=json.loads(json.dumps({'layout':layout,'mask':mask,'parameters':params.tolist(),'features':x.tolist()}))
        reloaded=kernel_matrix(np.asarray(serialized['features']),serialized['layout'],serialized['mask'],serialized['parameters'])
        reload_error=float(np.max(np.abs(k-reloaded)))
        hash_before=hashlib.sha256(qc.qcis(qc.overlap_gates(x[0],x[1],layout,mask,params),qubits).encode()).hexdigest()
        hash_after=hashlib.sha256(qc.qcis(qc.overlap_gates(serialized['features'][0],serialized['features'][1],serialized['layout'],serialized['mask'],serialized['parameters']),qubits).encode()).hexdigest()
        updated=kernel_matrix(x,layout,mask,params+1e-4*gradient)
        updated_value=float((updated*target).sum()/(np.linalg.norm(updated)*np.linalg.norm(target)))
        gradient_norm=float(np.linalg.norm(gradient))
        assert max(state_errors)<1e-10 and max(probability_errors)<1e-10
        assert max(fd_errors)<1e-7 and objective_error<1e-7
        assert abs(value-manual)<1e-12 and reload_error==0 and hash_before==hash_after
        assert updated_value>=value-1e-12
        if depth==1:
            assert gradient_norm<1e-12
        else:
            assert gradient_norm>1e-5
        cases.append({'qubits':qubits,'features':qubits,'depth':depth,'pairs':6,
                      'physical_occurrences_per_parameter':counts[0],
                      'max_state_error_internal_vs_qcis':max(state_errors),
                      'max_probability_error_inner_product_vs_qcis':max(probability_errors),
                      'max_probability_derivative_error_shift_vs_central_fd':max(fd_errors),
                      'kta_manual_definition_error':abs(value-manual),
                      'kta_chain_gradient_max_error_vs_central_fd':objective_error,
                      'parameter_serialization_kernel_error':reload_error,'qcis_hash_matches_reload':hash_before==hash_after,
                      'finite_difference_epsilon':epsilon,'analytic_kta_before':value,
                      'analytic_kta_after_small_ascent_step':updated_value,
                      'kta_gradient_l2_norm':gradient_norm,
                      'depth_one_shared_parameter_invariance_expected':depth==1,
                      'ascent_step_for_check_only':1e-4,'passed':True})
    result={'schema':'compact_pilot_v3_numerical_checks','execution_location':'local_cpu',
            'completed_utc':datetime.now(timezone.utc).isoformat(),'execution_kind':'analytic_constructed_fixture',
            'cloud_submissions':0,'actual_cloud_shots':0,'financial_data_used':False,
            'source_sha256':{'numerical_checks.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'study.py':hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
                             'circuits.py':hashlib.sha256(Path(qc.__file__).read_bytes()).hexdigest(),
                             'search.py':hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest()},
            'tolerances':{'state_and_probability_absolute':1e-10,'central_difference_gradient_absolute':1e-7,
                          'kta_definition_absolute':1e-12,'reload_kernel_absolute':0},
            'cases':cases,'all_passed':True,
            'limitations':['Analytic fixtures only; not financial prediction or finite-shot convergence evidence',
                           'Small ascent step validates gradient direction, not the registered 0.2 learning rate effectiveness',
                           'Single feature block at depth one has exact shared-parameter cancellation; finite-shot nonzero gradients can be sampling noise',
                           'No platform output, hardware noise or runtime capacity was checked']}
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional new private JSON output; default prints results only')
    args=parser.parse_args(argv)
    output=None
    if args.output:
        # The same read-only source guard forbids network/course-data output.
        from cloud_v3.sources import validate_output_path
        output=validate_output_path(args.output)
        if output.exists():
            parser.error('Output already exists; preserve prior numerical evidence')
    result=run_checks()
    payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf-8') as stream:
            stream.write(payload)
    print(payload,end='')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
