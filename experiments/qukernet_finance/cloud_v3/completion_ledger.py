"""Explicit evaluation-only continuation of a cleanly stopped compact pilot.

The parent window, deadline and entries are immutable.  No network or credential
handling is performed here.  A single child may spend only the frozen, never-
reserved evaluation operations and shares the parent's original resource caps.
"""
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

from cloud_v2.cloud import TianyanCloudAdapter as LegacyAdapter
from cloud_v2.preflight import PreflightBlocked
from cloud_v3 import ledger as compact

LedgerError = compact.LedgerError
BudgetExceeded = compact.BudgetExceeded
UnresolvedTask = compact.UnresolvedTask
DuplicateSubmission = compact.DuplicateSubmission
TERMINAL = compact.TERMINAL
digest = compact.digest
request_identity = compact.request_identity
atomic_json = compact.atomic_json
file_lock = compact.file_lock
utc = compact.utc
WINDOW_KIND = 'compact_quota_completion_v1'
PROTOCOL_ID = 'qukernet-50stocks-3h-cutoff20260325-qlib-missing-v1'
POLICY = {k:v for k,v in compact.MAIN_POLICY.items() if k != 'seconds'}
POLICY.update(mode='quota_completion', global_deadline_removed=True,
              budget_scope='parent_and_single_continuation')
AUTH_LIMITS = {'backend':'tianyan_sw','qubits':8,'shots_per_circuit':512,
    'parent_and_continuation_circuits_max':5292,
    'parent_and_continuation_requested_shots_max':2709504,
    'parent_and_continuation_batches_max':271,'circuits_per_batch_max':20,
    'batch_observation_seconds_max':300,'optimization_seconds':1800,
    'automatic_retries':False,'stop_on_unknown_failure_or_invalid_result':True,
    'max_compute_threads':8,'max_process_tree_memory_gib':32,'private_cache_gib':10}


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _epoch(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise LedgerError('Authorization time must include timezone')
    return parsed.timestamp()


def _parent(snapshot,parent_window_id):
    found=[w for w in snapshot.get('execution_windows',[]) if w['window_id']==parent_window_id]
    if len(found)!=1:raise LedgerError('One explicit parent window required')
    parent=found[0]
    entries=[e for e in snapshot['entries'] if e.get('execution_window_id')==parent_window_id]
    if (parent.get('kind')!=compact.WINDOW_KIND or parent.get('state')!='closed'
            or parent.get('outcome')!='operator_stop' or parent.get('stop_requested') is not True
            or parent.get('client_execution_stopped') is not True or not entries
            or any(e['state']!='validated' for e in entries)):
        raise UnresolvedTask('Parent must be a clean explicit operator stop with only validated attempts')
    return parent,entries


def _validate_operations(operations,parent_entries):
    if not isinstance(operations,dict) or not operations or len(operations)>120:
        raise LedgerError('Nonempty frozen remaining evaluation whitelist required')
    existing={e.get('operation_id') for e in parent_entries}
    for op,value in operations.items():
        if not isinstance(op,str) or not op.startswith('compact_pilot_v3/'+PROTOCOL_ID+'/'):
            raise LedgerError('Evaluation operation namespace changed')
        match=re.search(r'/(qukernet|random)/(\d+)/(endpoint/test|initial/test)/batch(0|[1-9]\d*)$',op)
        if (not match or int(match[4])>=40 or (match[1]=='random' and match[3]=='initial/test')
                or op in existing):
            raise DuplicateSubmission('Only never-reserved original evaluation operations are permitted')
        if (not isinstance(value,dict) or not _hash(value.get('request_identity'))
                or type(value.get('circuits')) is not int or not 1<=value['circuits']<=20
                or value.get('shots_per_circuit')!=512 or value.get('backend')!='tianyan_sw'
                or value.get('qubits')!=8):
            raise LedgerError('Malformed frozen evaluation request')
    if (len(parent_entries)+len(operations)>271
            or sum(e['circuits'] for e in parent_entries)+sum(v['circuits'] for v in operations.values())>5292):
        raise BudgetExceeded('Whitelist exceeds the original combined quota')


def _validate_amendment(amendment,snapshot,operations):
    if (not isinstance(amendment,dict) or amendment.get('schema')!='compact_quota_completion_amendment_v1'
            or amendment.get('mode')!='quota_completion' or amendment.get('policy')!=POLICY
            or amendment.get('account_scope_sha256')!=snapshot['account_scope_sha256']
            or not _hash(amendment.get('parent_ledger_sha256'))
            or not _hash(amendment.get('authorization_sha256'))):
        raise LedgerError('Invalid explicit continuation amendment')
    parent,entries=_parent(snapshot,amendment['parent_window_id'])
    _validate_operations(operations,entries)
    auth=amendment.get('authorization',{})
    if (auth.get('schema')!='qukernet_user_authorization_amendment_v1'
            or auth.get('authorization_source')!='explicit_user_message_in_current_thread'
            or auth.get('parent_window_id')!=parent['window_id']
            or auth.get('protocol_sha256')!=parent['protocol_sha256']
            or auth.get('limits_unchanged')!=AUTH_LIMITS
            or not _hash(auth.get('original_manifest_sha256'))
            or not isinstance(auth.get('user_request'),str) or not auth['user_request'].strip()
            or not isinstance(auth.get('authorized_change'),str) or not auth['authorized_change'].strip()
            or amendment.get('authorization_content_sha256')!=digest(auth)
            or amendment.get('parent_window_sha256')!=digest(parent)
            or amendment.get('parent_entries_sha256')!=digest(entries)
            or amendment.get('allowed_operations_sha256')!=digest(operations)
            or amendment.get('original_started_epoch')!=parent['started_epoch']
            or amendment.get('original_deadline_epoch')!=parent['deadline_epoch']):
        raise LedgerError('Continuation does not preserve its parent and user authorization')
    try:
        if (abs(_epoch(auth['original_first_reservation_utc'])-parent['started_epoch'])>1e-6
                or abs(_epoch(auth['original_global_deadline_utc'])-parent['deadline_epoch'])>1e-6):
            raise LedgerError('Original parent times differ')
        _epoch(auth['recorded_at_utc'])
    except (ValueError,TypeError,KeyError) as exc:
        raise LedgerError('Malformed authorization times') from exc
    return parent,entries


def build_amendment(snapshot,*,parent_window_id,parent_ledger_sha256,authorization,
                    authorization_sha256,allowed_operations):
    parent,entries=_parent(snapshot,parent_window_id)
    value={'schema':'compact_quota_completion_amendment_v1','mode':'quota_completion',
        'policy':dict(POLICY),'parent_window_id':parent_window_id,
        'account_scope_sha256':snapshot['account_scope_sha256'],
        'parent_ledger_sha256':parent_ledger_sha256,'parent_window_sha256':digest(parent),
        'parent_entries_sha256':digest(entries),'original_started_epoch':parent['started_epoch'],
        'original_deadline_epoch':parent['deadline_epoch'],
        'authorization':copy.deepcopy(authorization),'authorization_sha256':authorization_sha256,
        'authorization_content_sha256':digest(authorization),
        'allowed_operations_sha256':digest(allowed_operations)}
    _validate_amendment(value,snapshot,allowed_operations)
    return value


class CompletionLedger(compact.PersistentLedger):
    """Canonical account ledger, with one audited child and unchanged old windows."""
    def _read(self):
        value=json.loads(self.path.read_text(encoding='utf-8'))
        if (value.get('schema_version')!=1 or value.get('account_scope_sha256')!=self.account_hash
                or value.get('limits')!=self.limits or not isinstance(value.get('entries'),list)
                or value.get('deadline_epoch')!=value.get('started_epoch',0)+self.limits['seconds']):
            raise LedgerError('Legacy account schema or clock changed')
        entries=value['entries'];ids=[e.get('attempt_id') for e in entries]
        if len(ids)!=len(set(ids)) or any(not isinstance(x,str) for x in ids):
            raise LedgerError('Malformed or duplicate account attempt IDs')
        for entry in entries:
            if (type(entry.get('circuits')) is not int or entry['circuits']<=0
                    or type(entry.get('shots_per_circuit')) is not int or entry['shots_per_circuit']<=0
                    or entry.get('requested_shots')!=entry['circuits']*entry['shots_per_circuit']):
                raise LedgerError('Malformed account reservation counters')
        windows=value.get('execution_windows',[])
        if not isinstance(windows,list) or len({w.get('window_id') for w in windows})!=len(windows):
            raise LedgerError('Malformed account windows')
        children=[]
        for window in windows:
            kind=window.get('kind')
            if kind==WINDOW_KIND:children.append(window)
            elif kind==compact.WINDOW_KIND:self._validate_v3_window(window,entries)
            elif kind=='main_protocol_v1':compact.legacy.PersistentLedger._validate_main_window(window,entries)
            else:self._validate_legacy_preflight(window,entries)
        parents=[w.get('parent_window_id') for w in children]
        if len(parents)!=len(set(parents)):raise LedgerError('Parent may have only one continuation')
        for window in children:self._validate_child(value,window)
        return value

    def _validate_child(self,snapshot,window):
        if (window.get('policy')!=POLICY or window.get('mode')!='quota_completion'
                or window.get('state') not in {'authorized','running','closed'}
                or type(window.get('stop_requested')) is not bool or window.get('deadline_epoch') is not None
                or not _finite(window.get('authorized_epoch'))
                or abs(_epoch(window['authorized_utc'])-window['authorized_epoch'])>1e-6
                or not _hash(window.get('protocol_sha256'))
                or window.get('amendment_sha256')!=digest(window.get('amendment'))
                or window.get('allowed_operations_sha256')!=digest(window.get('allowed_operations'))):
            raise LedgerError('Continuation policy, authority or deadline changed')
        parent,prior=_validate_amendment(window['amendment'],snapshot,window['allowed_operations'])
        if (window.get('parent_window_id')!=parent['window_id']
                or window['protocol_sha256']!=parent['protocol_sha256']):
            raise LedgerError('Continuation parent/protocol binding changed')
        self._exact_remaining(prior,window['allowed_operations'])
        entries=[e for e in snapshot['entries'] if e.get('execution_window_id')==window['window_id']]
        if entries:
            if window['state']=='authorized' or window.get('started_epoch')!=entries[0]['reserved_epoch']:
                raise LedgerError('Continuation first reservation time changed')
        elif window['state']=='running' or window.get('started_epoch') is not None:
            raise LedgerError('Unstarted continuation has a running clock')
        seen=set()
        for entry in entries:
            metadata=entry.get('metadata',{});specs=metadata.get('circuits',[]);op=entry.get('operation_id')
            allowed=window['allowed_operations'].get(op)
            if (allowed is None or op in seen or entry.get('phase')!='main'
                    or metadata.get('context',{}).get('operation_id')!=op
                    or entry['circuits']!=allowed['circuits'] or entry['shots_per_circuit']!=512
                    or metadata.get('backend')!='tianyan_sw' or metadata.get('protocol_sha256')!=window['protocol_sha256']
                    or len(specs)!=entry['circuits'] or any(s.get('qubits')!=8 for s in specs)
                    or entry.get('request_identity')!=allowed['request_identity']
                    or entry.get('request_identity')!=request_identity(specs,512,'tianyan_sw','main')
                    or not _finite(entry.get('reserved_epoch'))
                    or entry['reserved_epoch']<window['authorized_epoch']
                    or entry.get('observation_deadline_epoch')!=entry['reserved_epoch']+300):
                raise LedgerError('Continuation reservation differs from frozen evaluation whitelist')
            seen.add(op)
        others=[e for e in snapshot['entries'] if e.get('execution_window_id')!=window['window_id']]
        if seen.intersection({e.get('operation_id') for e in others}):
            raise DuplicateSubmission('Continuation operation was already reserved elsewhere')
        combined=prior+entries
        if len(combined)>271 or sum(e['circuits'] for e in combined)>5292 or sum(e['requested_shots'] for e in combined)>2709504:
            raise BudgetExceeded('Combined parent and continuation quota exceeded')

    @staticmethod
    def _exact_remaining(parent_entries,operations):
        circuits=sum(e['circuits'] for e in parent_entries)+sum(v['circuits'] for v in operations.values())
        shots=sum(e['requested_shots'] for e in parent_entries)+sum(v['circuits']*v['shots_per_circuit'] for v in operations.values())
        if len(parent_entries)+len(operations)!=271 or circuits!=5292 or shots!=2709504:
            raise BudgetExceeded('Parent plus complete remaining whitelist must equal the original full quota')

    def authorize_main_window(self,*args,**kwargs):
        raise LedgerError('Only an explicit quota-completion amendment may create authority')

    def authorize_completion_window(self,window_id,protocol_sha256,authorization_note,amendment,allowed_operations):
        if (not isinstance(window_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',window_id)
                or not _hash(protocol_sha256) or not isinstance(authorization_note,str) or not authorization_note.strip()):
            raise LedgerError('Explicit continuation identity and authorization required')
        with file_lock(self.root/'ledger.lock'):
            snapshot=self._read();parent,entries=_validate_amendment(amendment,snapshot,allowed_operations)
            if hashlib.sha256(self.path.read_bytes()).hexdigest()!=amendment['parent_ledger_sha256']:
                raise LedgerError('Account ledger changed after continuation review')
            if protocol_sha256!=parent['protocol_sha256']:raise LedgerError('Parent protocol must be retained')
            if any(w['state']!='closed' for w in snapshot['execution_windows']):
                raise UnresolvedTask('All prior windows must be closed')
            if any(e['state'] not in TERMINAL for e in snapshot['entries']):
                raise UnresolvedTask('Unresolved prior attempt forbids continuation')
            if any(w['window_id']==window_id or w.get('parent_window_id')==parent['window_id'] for w in snapshot['execution_windows']):
                raise DuplicateSubmission('Continuation authorization is single use; closed children cannot reopen')
            if set(allowed_operations).intersection({e.get('operation_id') for e in snapshot['entries']}):
                raise DuplicateSubmission('Whitelist includes an existing account operation')
            self._exact_remaining(entries,allowed_operations)
            now=self.clock()
            if _epoch(amendment['authorization']['recorded_at_utc'])>now:
                raise LedgerError('User authorization record is dated after activation')
            window={'kind':WINDOW_KIND,'mode':'quota_completion','window_id':window_id,
                'parent_window_id':parent['window_id'],'protocol_sha256':protocol_sha256,
                'authorization_note':authorization_note,'policy':dict(POLICY),
                'amendment':copy.deepcopy(amendment),'amendment_sha256':digest(amendment),
                'allowed_operations':copy.deepcopy(allowed_operations),'allowed_operations_sha256':digest(allowed_operations),
                'state':'authorized','authorized_utc':utc(now),'authorized_epoch':now,'started_epoch':None,
                'deadline_epoch':None,'stop_requested':False}
            snapshot['execution_windows'].append(window);atomic_json(self.path,snapshot);return window

    def effective_deadline(self,window_id=None):
        if window_id is None:raise LedgerError('Continuation requires its explicit window')
        window=self.window(window_id)
        if window.get('kind')!=WINDOW_KIND:raise LedgerError('Parent authority is sealed')
        if window['state']=='closed':raise DuplicateSubmission('Closed continuation cannot submit or poll')
        return math.inf  # Deliberately never serialized as a JSON deadline.

    def reserve(self,circuits,shots,phase,fingerprint,metadata,window_id=None):
        if type(circuits) is not int or not 1<=circuits<=20 or type(shots) is not int or shots!=512 or phase!='main':
            raise BudgetExceeded('Continuation permits 1..20 circuits of 512 shots only')
        with file_lock(self.root/'ledger.lock'):
            snapshot=self._read();found=[w for w in snapshot['execution_windows'] if w['window_id']==window_id]
            if len(found)!=1 or found[0].get('kind')!=WINDOW_KIND:raise LedgerError('Explicit continuation authority required')
            window=found[0]
            if window['state'] not in {'authorized','running'} or window['stop_requested']:
                raise DuplicateSubmission('Continuation is stopped')
            op=metadata.get('context',{}).get('operation_id');specs=metadata.get('circuits',[])
            allowed=window['allowed_operations'].get(op)
            identity=request_identity(specs,shots,'tianyan_sw',phase)
            if (allowed is None or allowed['circuits']!=circuits or allowed['request_identity']!=identity
                    or metadata.get('backend')!='tianyan_sw' or metadata.get('protocol_sha256')!=window['protocol_sha256']
                    or len(specs)!=circuits or any(s.get('qubits')!=8 for s in specs)):
                raise BudgetExceeded('Request is not the frozen remaining evaluation operation')
            if any(e.get('operation_id')==op or e.get('fingerprint')==fingerprint for e in snapshot['entries']):
                raise DuplicateSubmission('Reserved operations may never be resubmitted')
            if any(e['state'] not in TERMINAL for e in snapshot['entries']):raise UnresolvedTask('Prior unresolved attempt')
            combined=[e for e in snapshot['entries'] if e.get('execution_window_id') in {window_id,window['parent_window_id']}]
            if (len(combined)+1>271 or sum(e['circuits'] for e in combined)+circuits>5292
                    or sum(e['requested_shots'] for e in combined)+circuits*shots>2709504
                    or sum(e['circuits'] for e in snapshot['entries'])+circuits>self.limits['circuits']
                    or sum(e['requested_shots'] for e in snapshot['entries'])+circuits*shots>self.limits['shots']):
                raise BudgetExceeded('Original combined or account-wide quota exhausted')
            now=self.clock()
            if not _finite(now) or now<window['authorized_epoch']:
                raise LedgerError('Reservation precedes explicit continuation activation')
            if window['state']=='authorized':window.update(state='running',started_epoch=now,started_utc=utc(now))
            entry={'attempt_id':uuid.uuid4().hex,'state':'submission_pending','submission_started_utc':utc(now),
                'phase':phase,'circuits':circuits,'shots_per_circuit':shots,'requested_shots':circuits*shots,
                'fingerprint':fingerprint,'metadata':metadata,'execution_window_id':window_id,
                'reserved_epoch':now,'observation_deadline_epoch':now+300,'operation_id':op,
                'request_identity':identity,'history':[{'state':'submission_pending','utc':utc(now)}]}
            snapshot['entries'].append(entry);atomic_json(self.path,snapshot);return entry

    def update(self,attempt_id,state,**fields):
        allowed={'submission_pending':{'submitted','submission_unknown','submission_rejected'},
                 'submitted':{'submitted','validated','terminal_failure','observation_unknown','received_invalid'}}
        with file_lock(self.root/'ledger.lock'):
            data=self._read();entry=next((e for e in data['entries'] if e['attempt_id']==attempt_id),None)
            if entry is None:raise LedgerError('Attempt not found')
            window=next(w for w in data['execution_windows'] if w['window_id']==entry.get('execution_window_id'))
            if window.get('kind')!=WINDOW_KIND:raise LedgerError('Historical parent and entries are sealed')
            if state not in allowed.get(entry['state'],set()):raise LedgerError('Forbidden state transition')
            if state=='validated' and (window['state']=='closed' or self.clock()>=entry['observation_deadline_epoch']):
                raise BudgetExceeded('No late validation after the persisted batch deadline')
            protected={'attempt_id','fingerprint','circuits','shots_per_circuit','requested_shots','submission_started_utc',
                'phase','history','metadata','state','execution_window_id','reserved_epoch','observation_deadline_epoch','operation_id','request_identity'}
            if protected.intersection(fields):raise LedgerError('Frozen reservation fields cannot change')
            if 'job_ids' in fields:
                ids=fields['job_ids'];other={jid for e in data['entries'] if e is not entry for jid in e.get('job_ids',[])}
                if len(ids)!=len(set(ids)) or other.intersection(ids):raise LedgerError('Task IDs reused')
            entry.update(fields);entry['state']=state;entry['history'].append({'state':state,'utc':utc(self.clock())})
            if state not in {'submission_pending','submitted'} and (state!='validated' or window['stop_requested']):
                window.update(state='closed',outcome='operator_stop' if state=='validated' else state,
                    closed_utc=utc(self.clock()),client_execution_stopped=True,backend_cancelled=False,stop_requested=True)
            atomic_json(self.path,data)

    def close_main_window(self,window_id,outcome,reason):
        if not isinstance(reason,str) or not reason.strip():raise LedgerError('Closure reason required')
        with file_lock(self.root/'ledger.lock'):
            data=self._read();window=next(w for w in data['execution_windows'] if w['window_id']==window_id)
            if window.get('kind')!=WINDOW_KIND:raise LedgerError('Historical parent is sealed')
            if window['state']=='closed':return window
            entries=[e for e in data['entries'] if e.get('execution_window_id')==window_id]
            if outcome=='completed' and (not entries or any(e['state']!='validated' for e in entries)
                    or {e['operation_id'] for e in entries}!=set(window['allowed_operations'])):
                raise UnresolvedTask('Cannot complete until every frozen remaining operation is validated')
            window.update(state='closed',outcome=outcome,reason=reason,closed_utc=utc(self.clock()),
                stop_requested=True,client_execution_stopped=True,backend_cancelled=False)
            atomic_json(self.path,data);return window

    def request_stop(self,window_id,reason):
        if not isinstance(reason,str) or not reason.strip():raise LedgerError('Explicit stop reason required')
        with file_lock(self.root/'ledger.lock'):
            data=self._read();window=next(w for w in data['execution_windows'] if w['window_id']==window_id)
            if window.get('kind')!=WINDOW_KIND:raise LedgerError('Historical parent is sealed')
            if window['state']=='closed':return window
            window.update(stop_requested=True,stop_reason=reason,stop_requested_utc=utc(self.clock()))
            if not any(e.get('execution_window_id')==window_id and e['state'] in {'submission_pending','submitted'} for e in data['entries']):
                window.update(state='closed',outcome='operator_stop',reason=reason,closed_utc=utc(self.clock()),
                    client_execution_stopped=True,backend_cancelled=False)
            atomic_json(self.path,data);return window


class CompletionAdapter(LegacyAdapter):
    """Use the reviewed live SDK checks/transport with continuation authority."""
    def __init__(self,transport,ledger,evidence=None,**kwargs):
        wid=kwargs.get('execution_window_id');window=ledger.window(wid) if wid else None
        if (not isinstance(ledger,CompletionLedger) or window is None or window.get('kind')!=WINDOW_KIND
                or kwargs.get('execution_protocol_sha256')!=window['protocol_sha256']
                or kwargs.get('backend','tianyan_sw')!='tianyan_sw'):
            raise PreflightBlocked('Matching continuation ledger, protocol and backend required')
        validator=kwargs.get('result_validator');kwargs['result_validator']=validator or (lambda values:None)
        super().__init__(transport,ledger,evidence,**kwargs)
        self.main_window=True;self.result_validator=validator

    def execution_identity(self):
        result=super().execution_identity();window=self.ledger.window(self.execution_window_id)
        result.update(execution_window_kind=WINDOW_KIND,completion_policy=dict(POLICY),
            parent_window_id=window['parent_window_id'],amendment_sha256=window['amendment_sha256'],
            completion_ledger_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result
