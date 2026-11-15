"""Compact pilot ledger: append a three-hour authority to the canonical account.

Historical v2 entries and windows retain their original validation and counters.
All new pilot requests, including the four analytic probes, use phase='main';
their context records the analytic/main role. No network or credentials here.
"""
import hashlib
import json
from pathlib import Path
import uuid

from cloud_v2 import ledger as legacy
from cloud_v2.cloud import TianyanCloudAdapter as LegacyAdapter
from cloud_v2.preflight import PreflightBlocked

LedgerError = legacy.LedgerError
BudgetExceeded = legacy.BudgetExceeded
UnresolvedTask = legacy.UnresolvedTask
DuplicateSubmission = legacy.DuplicateSubmission
atomic_json = legacy.atomic_json
private_directory = legacy.private_directory
file_lock = legacy.file_lock
request_identity = legacy.request_identity
utc = legacy.utc
digest = legacy.digest
TERMINAL = legacy.TERMINAL
WINDOW_KIND = 'compact_pilot_v3'
MAIN_POLICY = {'backend':'tianyan_sw', 'phase':'main', 'circuits':5292,
               'shots':2709504, 'shots_per_circuit':512, 'qubits':8,
               'max_attempts':271, 'batch':20, 'seconds':3*3600,
               'observation_seconds':300}


def _valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


class PersistentLedger(legacy.PersistentLedger):
    """Same account schema; a distinct window kind controls all new requests."""
    def _read(self):
        value=json.loads(self.path.read_text(encoding='utf-8'))
        if (value.get('schema_version')!=1 or value.get('account_scope_sha256')!=self.account_hash
                or value.get('limits')!=self.limits or not isinstance(value.get('entries'),list)):
            raise LedgerError('Ledger schema, account or frozen limits mismatch')
        if value.get('deadline_epoch')!=value.get('started_epoch',0)+self.limits['seconds']:
            raise LedgerError('Legacy account deadline changed')
        ids=[e.get('attempt_id') for e in value['entries']]
        if len(ids)!=len(set(ids)) or any(not isinstance(i,str) for i in ids):
            raise LedgerError('Duplicate or malformed ledger attempts')
        for e in value['entries']:
            if (type(e.get('circuits')) is not int or e['circuits']<=0
                    or type(e.get('shots_per_circuit')) is not int or e['shots_per_circuit']<=0
                    or e.get('requested_shots')!=e['circuits']*e['shots_per_circuit']):
                raise LedgerError('Malformed reserved resource counters')
        windows=value.get('execution_windows',[])
        if not isinstance(windows,list) or len({w.get('window_id') for w in windows})!=len(windows):
            raise LedgerError('Malformed execution windows')
        for w in windows:
            if w.get('kind')==WINDOW_KIND:
                self._validate_v3_window(w,value['entries'])
            elif w.get('kind')=='main_protocol_v1':
                legacy.PersistentLedger._validate_main_window(w,value['entries'])
            else:
                self._validate_legacy_preflight(w,value['entries'])
        return value

    @staticmethod
    def _validate_legacy_preflight(w,entries):
        # Identical policy/clock checks to the unchanged v2 implementation.
        if (w.get('policy')!={'backend':'tianyan_sw','phase':'preflight','circuits':12,
                'shots_per_circuit':128,'qubits':8,'max_attempts':1,'seconds':72*3600,
                'observation_seconds':300} or w.get('state') not in {'authorized','reserved','closed'}
                or not isinstance(w.get('window_id'),str) or not w['window_id']):
            raise LedgerError('Legacy execution window policy mismatch')
        related=[e for e in entries if e.get('execution_window_id')==w['window_id']]
        unstarted_closed=w['state']=='closed' and w.get('outcome')=='preflight_blocked_before_reservation'
        if w['state']=='authorized' or unstarted_closed:
            if related or w.get('started_epoch') is not None or w.get('deadline_epoch') is not None or w.get('attempt_id') is not None:
                raise LedgerError('Unstarted legacy window has running clock or attempts')
        elif (len(related)!=1 or w.get('attempt_id')!=related[0]['attempt_id']
              or w.get('deadline_epoch')!=w.get('started_epoch',0)+w['policy']['seconds']
              or related[0].get('reserved_epoch')!=w['started_epoch']
              or related[0].get('observation_deadline_epoch')!=w['started_epoch']+300):
            raise LedgerError('Legacy execution window clock or attempt changed')

    @staticmethod
    def _validate_v3_window(w,entries):
        if (w.get('policy')!=MAIN_POLICY or w.get('state') not in {'authorized','running','closed'}
                or not isinstance(w.get('window_id'),str) or not w['window_id']
                or not _valid_hash(w.get('protocol_sha256')) or type(w.get('stop_requested')) is not bool):
            raise LedgerError('Compact policy, protocol, or stop marker changed')
        related=[e for e in entries if e.get('execution_window_id')==w['window_id']]
        if not related:
            if w['state']=='running' or w.get('started_epoch') is not None or w.get('deadline_epoch') is not None:
                raise LedgerError('Unstarted compact window has running clock')
            return
        if (w['state']=='authorized' or w.get('started_epoch')!=related[0].get('reserved_epoch')
                or w.get('deadline_epoch')!=w['started_epoch']+MAIN_POLICY['seconds']):
            raise LedgerError('Compact persisted three-hour deadline changed')
        operations=[]
        for e in related:
            m=e.get('metadata',{})
            if (e.get('phase')!='main' or e['shots_per_circuit']!=512 or not 1<=e['circuits']<=20
                    or m.get('backend')!='tianyan_sw' or m.get('protocol_sha256')!=w['protocol_sha256']
                    or len(m.get('circuits',[]))!=e['circuits'] or any(s.get('qubits')!=8 for s in m['circuits'])
                    or not isinstance(e.get('operation_id'),str) or not e['operation_id'].strip()
                    or e.get('observation_deadline_epoch')!=min(e.get('reserved_epoch',0)+300,w['deadline_epoch'])
                    or e['reserved_epoch']>=w['deadline_epoch']
                    or e.get('request_identity')!=request_identity(m['circuits'],512,'tianyan_sw','main')):
                raise LedgerError('Compact reservation constraints changed')
            operations.append(e['operation_id'])
        if (len(set(operations))!=len(operations) or len(related)>271
                or sum(e['circuits'] for e in related)>5292
                or sum(e['requested_shots'] for e in related)>2709504):
            raise LedgerError('Compact resource limits exceeded or operation repeated')

    def authorize_main_window(self,window_id,protocol_sha256,authorization_note):
        if (not isinstance(window_id,str) or not window_id.strip() or not _valid_hash(protocol_sha256)
                or not isinstance(authorization_note,str) or not authorization_note.strip()):
            raise LedgerError('Explicit authorization and protocol hash required')
        declaration={'kind':WINDOW_KIND,'window_id':window_id,'protocol_sha256':protocol_sha256,
                     'authorization_note':authorization_note,'policy':dict(MAIN_POLICY)}
        with file_lock(self.root/'ledger.lock'):
            data=self._read();windows=data.setdefault('execution_windows',[])
            same=[w for w in windows if w['window_id']==window_id]
            if same:
                if any(same[0].get(k)!=v for k,v in declaration.items()):
                    raise LedgerError('Existing authorization cannot be changed')
                return same[0]
            if any(w['state']!='closed' for w in windows):
                raise UnresolvedTask('Earlier window is still open')
            if any(e['state'] not in TERMINAL for e in data['entries']):
                raise UnresolvedTask('Historical unresolved reservations block authorization')
            w={**declaration,'state':'authorized','authorized_utc':utc(self.clock()),
               'started_epoch':None,'deadline_epoch':None,'stop_requested':False}
            windows.append(w);atomic_json(self.path,data);return w

    def authorize_window(self,*args,**kwargs):
        raise LedgerError('Use the explicit compact authority; legacy windows are read-only')

    def close_main_window(self,window_id,outcome,reason):
        if self.window(window_id).get('kind')!=WINDOW_KIND:
            return super().close_main_window(window_id,outcome,reason)
        if not isinstance(outcome,str) or not outcome or not isinstance(reason,str) or not reason:
            raise LedgerError('Closure outcome and reason required')
        with file_lock(self.root/'ledger.lock'):
            data=self._read();w=next(w for w in data['execution_windows'] if w['window_id']==window_id)
            if w['state']=='closed':return w
            related=[e for e in data['entries'] if e.get('execution_window_id')==window_id]
            if outcome=='completed' and (not related or any(e['state']!='validated' for e in related)):
                raise UnresolvedTask('Cannot complete an unstarted or unresolved window')
            w.update(state='closed',outcome=outcome,reason=reason,closed_utc=utc(self.clock()),
                     stop_requested=True,client_execution_stopped=True,backend_cancelled=False)
            atomic_json(self.path,data);return w

    def request_stop(self,window_id,reason):
        """Persist a boundary stop; let the current submitted batch finish once."""
        if not isinstance(reason,str) or not reason.strip():raise LedgerError('Stop reason required')
        with file_lock(self.root/'ledger.lock'):
            data=self._read();found=[w for w in data.get('execution_windows',[]) if w['window_id']==window_id]
            if len(found)!=1 or found[0].get('kind')!=WINDOW_KIND:raise LedgerError('Compact window not found')
            w=found[0]
            if w['state']=='closed':return w
            w.update(stop_requested=True,stop_reason=reason,stop_requested_utc=utc(self.clock()))
            active=[e for e in data['entries'] if e.get('execution_window_id')==window_id and e['state'] in {'submission_pending','submitted'}]
            if not active:
                w.update(state='closed',outcome='operator_stop',reason=reason,closed_utc=utc(self.clock()),
                         client_execution_stopped=True,backend_cancelled=False)
            atomic_json(self.path,data);return w

    def effective_deadline(self,window_id=None):
        if window_id is None:return super().effective_deadline(None)
        w=self.window(window_id)
        if w.get('kind')!=WINDOW_KIND:return super().effective_deadline(window_id)
        if w['state']=='closed':raise DuplicateSubmission('Compact window closed; no submit or poll')
        return w['deadline_epoch']

    def reserve(self,circuits,shots,phase,fingerprint,metadata,window_id=None):
        if type(circuits) is not int or not 1<=circuits<=min(20,self.limits['batch']):
            raise BudgetExceeded('Batch must contain1..20 circuits')
        if type(shots) is not int or shots!=512 or phase!='main':
            raise BudgetExceeded('Compact requests require512 shots and main accounting phase')
        with file_lock(self.root/'ledger.lock'):
            data=self._read();now=self.clock()
            found=[w for w in data.get('execution_windows',[]) if w['window_id']==window_id]
            if len(found)!=1 or found[0].get('kind')!=WINDOW_KIND:raise LedgerError('Compact authority required')
            w=found[0]
            if w['state'] not in {'authorized','running'} or w['stop_requested']:
                raise DuplicateSubmission('Compact window stopped; no new reservation')
            op=metadata.get('context',{}).get('operation_id')
            specs=metadata.get('circuits',[])
            if (metadata.get('backend')!='tianyan_sw' or metadata.get('protocol_sha256')!=w['protocol_sha256']
                    or len(specs)!=circuits or any(s.get('qubits')!=8 for s in specs)
                    or not isinstance(op,str) or not op.strip()):
                raise BudgetExceeded('Request differs from frozen compact protocol')
            identity=request_identity(specs,shots,'tianyan_sw',phase)
            if any(e.get('operation_id')==op for e in data['entries']):
                raise DuplicateSubmission('Logical operation already reserved; no replay')
            if any(e['state'] not in TERMINAL for e in data['entries']):
                raise UnresolvedTask('Unresolved prior attempt blocks submission')
            if any(e.get('fingerprint')==fingerprint for e in data['entries']):
                raise DuplicateSubmission('Fingerprint already reserved')
            prior=[e for e in data['entries'] if e.get('execution_window_id')==window_id]
            if (len(prior)+1>271 or sum(e['circuits'] for e in prior)+circuits>5292
                    or sum(e['requested_shots'] for e in prior)+circuits*shots>2709504):
                raise BudgetExceeded('Compact reservation quota exceeded')
            if w['deadline_epoch'] is not None and now>=w['deadline_epoch']:
                raise BudgetExceeded('Three-hour window expired')
            if (sum(e['circuits'] for e in data['entries'])+circuits>self.limits['circuits']
                    or sum(e['requested_shots'] for e in data['entries'])+circuits*shots>self.limits['shots']):
                raise BudgetExceeded('Account-wide reservation quota exceeded')
            if w['state']=='authorized':
                w.update(state='running',started_epoch=now,started_utc=utc(now),deadline_epoch=now+10800)
            e={'attempt_id':uuid.uuid4().hex,'state':'submission_pending','submission_started_utc':utc(now),
               'phase':phase,'circuits':circuits,'shots_per_circuit':shots,'requested_shots':circuits*shots,
               'fingerprint':fingerprint,'metadata':metadata,'execution_window_id':window_id,
               'reserved_epoch':now,'observation_deadline_epoch':min(now+300,w['deadline_epoch']),
               'operation_id':op,'request_identity':identity,'history':[{'state':'submission_pending','utc':utc(now)}]}
            data['entries'].append(e);atomic_json(self.path,data);return e

    def update(self,attempt_id,state,**fields):
        snap=self.snapshot();entry=next((e for e in snap['entries'] if e['attempt_id']==attempt_id),None)
        if entry is None:raise LedgerError('Attempt missing')
        bound=next((w for w in snap.get('execution_windows',[]) if w['window_id']==entry.get('execution_window_id')),None)
        if bound is None or bound.get('kind')!=WINDOW_KIND:return super().update(attempt_id,state,**fields)
        allowed={'submission_pending':{'submitted','submission_unknown','submission_rejected'},
                 'submitted':{'submitted','validated','terminal_failure','observation_unknown','received_invalid'}}
        with file_lock(self.root/'ledger.lock'):
            data=self._read();e=next(e for e in data['entries'] if e['attempt_id']==attempt_id)
            w=next(w for w in data['execution_windows'] if w['window_id']==e['execution_window_id'])
            if state not in allowed.get(e['state'],set()):raise LedgerError('Forbidden state transition')
            if state=='validated' and (w['state']=='closed' or self.clock()>=e['observation_deadline_epoch']):
                raise BudgetExceeded('No late validation or validation after closed window')
            protected={'attempt_id','fingerprint','circuits','shots_per_circuit','requested_shots','submission_started_utc',
                       'phase','history','metadata','state','execution_window_id','reserved_epoch',
                       'observation_deadline_epoch','operation_id','request_identity'}
            if protected.intersection(fields):raise LedgerError('Frozen reservation fields cannot change')
            if 'job_ids' in fields:
                ids=fields['job_ids']
                if len(ids)!=len(set(ids)):raise LedgerError('Duplicate task IDs in batch')
                prior={jid for other in data['entries'] if other is not e for jid in other.get('job_ids',[])}
                if prior.intersection(ids):raise LedgerError('Task ID reused across attempts')
            e.update(fields);e['state']=state;e['history'].append({'state':state,'utc':utc(self.clock())})
            if state not in {'submission_pending','submitted'} and (state!='validated' or w['stop_requested']):
                w.update(state='closed',outcome='operator_stop' if state=='validated' else state,
                         closed_utc=utc(self.clock()),client_execution_stopped=True,backend_cancelled=False,
                         stop_requested=True)
            atomic_json(self.path,data)


class TianyanCloudAdapter(LegacyAdapter):
    """Bind the unchanged reviewed transport/validation to the new policy kind."""
    def __init__(self,transport,ledger,evidence=None,**kwargs):
        wid=kwargs.get('execution_window_id');window=ledger.window(wid) if wid is not None else None
        if (not isinstance(ledger,PersistentLedger) or window is None or window.get('kind')!=WINDOW_KIND
                or kwargs.get('execution_protocol_sha256')!=window['protocol_sha256']
                or kwargs.get('backend','tianyan_sw')!=MAIN_POLICY['backend']):
            raise PreflightBlocked('Compact adapter requires matching compact ledger, authority, and protocol')
        validator=kwargs.get('result_validator')
        # v2's constructor sees an unfamiliar kind; satisfy its preflight-only
        # constructor guard, then retain the actual caller validator unchanged.
        kwargs['result_validator']=validator if validator is not None else (lambda results:None)
        super().__init__(transport,ledger,evidence,**kwargs)
        self.main_window=True
        self.result_validator=validator

    def execution_identity(self):
        result=super().execution_identity()
        result.update(execution_window_kind=WINDOW_KIND,compact_policy=dict(MAIN_POLICY),
                      compact_ledger_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result
