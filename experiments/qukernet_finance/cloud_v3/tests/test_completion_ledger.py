"""No cloud calls: explicit parent/child authorization, invariants and deadlines."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'cloud_v2/tests'))
from cloud_v3 import ledger as compact
from cloud_v3.completion_ledger import (CompletionLedger,CompletionAdapter,build_amendment,
    AUTH_LIMITS,POLICY,WINDOW_KIND,PROTOCOL_ID,LedgerError,BudgetExceeded,DuplicateSubmission,UnresolvedTask)
from cloud_v2.preflight import PreflightBlocked
from cloud_v2.cloud import UnknownSubmission,InvalidResult,circuit_spec
from test_authorized_window_integration import VirtualClock,twelve_specs,fixture_evidence
from test_main_window import MainTransport


class CompletionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.clock=VirtualClock();self.scope='completion-offline'
        self.parent=compact.PersistentLedger(self.temp.name,self.scope,clock=self.clock.wall)
        self.pid='parent';self.cid='child';self.protocol='d'*64
        self.parent.authorize_main_window(self.pid,self.protocol,'old fixture')
        specs=self.specs(4,'parent')
        entry=self.parent.reserve(4,512,'main','parent-fp',self.meta(specs,'parent-probe'),window_id=self.pid)
        self.finish(self.parent,entry);self.parent.request_stop(self.pid,'explicit clean pause for same-quota completion')
        # Realistic full-prefix fixture:151 legal batches /2908 circuits, leaving
        #120 evaluation batches /2384 circuits. No reduced production policy.
        snapshot=self.parent.snapshot();template=snapshot['entries'][0];snapshot['entries']=[]
        for index,count in enumerate([4]+[20]*144+[4]*6):
            entry=copy.deepcopy(template);op='finished-stage/'+str(index)
            specs=self.specs(count,op);entry.update(attempt_id='parent-'+str(index),circuits=count,
                requested_shots=count*512,operation_id=op,fingerprint='parent-fp-'+str(index),
                metadata=self.meta(specs,op),request_identity=compact.request_identity(specs,512,'tianyan_sw','main'),
                job_ids=['parent-job-'+str(index)+'-'+str(i) for i in range(count)])
            snapshot['entries'].append(entry)
        compact.atomic_json(self.parent.path,snapshot)
        self.parent_state=self.parent.snapshot();self.before=self.parent.path.read_bytes()
        self.op=self.operation('qukernet','endpoint/test',0)
        self.spec=self.specs(4,'child')
        self.allowed={}
        for arm,kind in [('qukernet','endpoint/test'),('random','endpoint/test'),('qukernet','initial/test')]:
            for index in range(40):
                op=self.operation(arm,kind,index)
                self.allowed[op]=self.allowed_value(self.spec if op==self.op else self.specs(20,op))
        self.authority=self.authorization(self.parent_state)
        self.ledger=CompletionLedger(self.temp.name,self.scope,clock=self.clock.wall)

    def tearDown(self):self.temp.cleanup()
    def operation(self,arm,kind,index):return f'compact_pilot_v3/{PROTOCOL_ID}/window/{arm}/20261115/{kind}/batch{index}'
    def specs(self,n,prefix):return [{'id':f'{prefix}-{i}','sha256':hashlib.sha256(f'{prefix}-{i}'.encode()).hexdigest(),'qubits':8} for i in range(n)]
    def meta(self,specs,op):return {'circuits':specs,'backend':'tianyan_sw','protocol_sha256':self.protocol,'context':{'operation_id':op}}
    def allowed_value(self,specs):return {'request_identity':compact.request_identity(specs,512,'tianyan_sw','main'),'circuits':len(specs),'shots_per_circuit':512,'backend':'tianyan_sw','qubits':8}
    def authorization(self,snapshot):
        w=next(w for w in snapshot['execution_windows'] if w['window_id']==self.pid)
        return {'schema':'qukernet_user_authorization_amendment_v1','authorization_source':'explicit_user_message_in_current_thread',
            'parent_window_id':self.pid,'protocol_sha256':self.protocol,'limits_unchanged':copy.deepcopy(AUTH_LIMITS),
            'original_manifest_sha256':'a'*64,'user_request':'Complete the existing fixed experiment.',
            'authorized_change':'Remove only global three-hour cutoff.','original_first_reservation_utc':compact.utc(w['started_epoch']),
            'original_global_deadline_utc':compact.utc(w['deadline_epoch']),'recorded_at_utc':compact.utc(self.clock.wall())}
    def amendment(self):return build_amendment(self.ledger.snapshot(),parent_window_id=self.pid,parent_ledger_sha256=hashlib.sha256(self.ledger.path.read_bytes()).hexdigest(),authorization=self.authority,authorization_sha256='b'*64,allowed_operations=self.allowed)
    def authorize(self):return self.ledger.authorize_completion_window(self.cid,self.protocol,'explicit evaluation-only continuation',self.amendment(),self.allowed)
    def reserve(self,op=None,specs=None,fp='child-fp'):
        op=self.op if op is None else op;specs=self.spec if specs is None else specs
        return self.ledger.reserve(len(specs),512,'main',fp,self.meta(specs,op),window_id=self.cid)
    def finish(self,ledger,entry):
        ledger.update(entry['attempt_id'],'submitted',job_ids=[entry['attempt_id']+'-'+str(i) for i in range(entry['circuits'])])
        ledger.update(entry['attempt_id'],'validated')

    def test_parent_original_bytes_and_fields_retained(self):
        self.authorize();self.finish(self.ledger,self.reserve());after=self.ledger.snapshot()
        self.assertEqual(after['entries'][:151],self.parent_state['entries'])
        self.assertEqual(after['execution_windows'][:1],self.parent_state['execution_windows'])
        for key in ('schema_version','limits','started_epoch','deadline_epoch'):self.assertEqual(after[key],self.parent_state[key])

    def test_expired_original_deadline_does_not_clip_new_batch(self):
        self.clock.advance(20000);self.authorize();entry=self.reserve()
        self.assertTrue(math.isinf(self.ledger.effective_deadline(self.cid)))
        self.assertEqual(entry['observation_deadline_epoch'],self.clock.wall()+300)
        self.assertIsNone(self.ledger.window(self.cid)['deadline_epoch'])
        self.assertNotIn('Infinity',self.ledger.path.read_text())
        self.finish(self.ledger,entry)

    def test_restart_preserves_batch_deadline_and_no_duplicate(self):
        self.authorize();entry=self.reserve();self.clock.advance(50)
        again=CompletionLedger(self.temp.name,self.scope,clock=self.clock.wall)
        self.assertEqual(again.snapshot()['entries'][-1]['observation_deadline_epoch'],entry['observation_deadline_epoch'])
        with self.assertRaises(DuplicateSubmission):again.reserve(4,512,'main','different',self.meta(self.spec,self.op),window_id=self.cid)

    def test_300_seconds_remains_hard(self):
        self.authorize();entry=self.reserve();self.ledger.update(entry['attempt_id'],'submitted',job_ids=['slow'])
        self.clock.advance(300)
        with self.assertRaises(BudgetExceeded):self.ledger.update(entry['attempt_id'],'validated')

    def test_no_parent_reopen_or_parent_update(self):
        self.authorize()
        with self.assertRaises(LedgerError):self.ledger.reserve(4,512,'main','reopen',self.meta(self.spec,self.op),window_id=self.pid)
        with self.assertRaises(LedgerError):self.ledger.close_main_window(self.pid,'completed','rewrite')
        with self.assertRaises(LedgerError):self.ledger.update(self.parent_state['entries'][0]['attempt_id'],'validated')

    def test_failed_parent_and_unknown_parent_forbidden(self):
        for outcome,state in [('client_stopped','validated'),('operator_stop','observation_unknown'),('completed','validated')]:
            value=copy.deepcopy(self.parent_state);value['execution_windows'][0]['outcome']=outcome;value['entries'][0]['state']=state
            compact.atomic_json(self.ledger.path,value)
            with self.assertRaises(UnresolvedTask):self.amendment()
        compact.atomic_json(self.ledger.path,self.parent_state)

    def test_unrelated_unknown_account_entry_forbidden(self):
        value=copy.deepcopy(self.parent_state);entry=copy.deepcopy(value['entries'][0]);entry['attempt_id']='other';entry.pop('execution_window_id');entry['state']='observation_unknown';value['entries'].append(entry)
        compact.atomic_json(self.ledger.path,value)
        with self.assertRaises(UnresolvedTask):self.authorize()

    def test_whitelist_rejects_training_random_initial_and_changed_namespace(self):
        for op in [self.operation('qukernet','labels/0',0),self.operation('random','initial/test',0),self.op.replace(PROTOCOL_ID,'another-version'),self.operation('qukernet','endpoint/test',40)]:
            self.allowed={op:self.allowed_value(self.spec)}
            with self.assertRaises(LedgerError):self.amendment()

    def test_request_content_shots_backend_and_protocol_bound(self):
        self.authorize()
        for field,value in [('backend','other'),('protocol_sha256','e'*64)]:
            meta=self.meta(self.spec,self.op);meta[field]=value
            with self.assertRaises(BudgetExceeded):self.ledger.reserve(4,512,'main','f',meta,window_id=self.cid)
        with self.assertRaises(BudgetExceeded):self.reserve(specs=self.specs(4,'changed'))
        with self.assertRaises(BudgetExceeded):self.ledger.reserve(4,128,'main','f',self.meta(self.spec,self.op),window_id=self.cid)

    def test_changed_authorization_or_parent_hash_rejected(self):
        for field in ['original_deadline_epoch','parent_window_sha256','parent_entries_sha256','allowed_operations_sha256']:
            amendment=self.amendment();amendment[field]='c'*64
            with self.assertRaises(LedgerError):self.ledger.authorize_completion_window(self.cid,self.protocol,'auth',amendment,self.allowed)
        amendment=self.amendment();amendment['authorization']['limits_unchanged']['automatic_retries']=True
        with self.assertRaises(LedgerError):self.ledger.authorize_completion_window(self.cid,self.protocol,'auth',amendment,self.allowed)

    def test_ledger_changed_after_review_rejected(self):
        amendment=self.amendment();state=self.ledger.snapshot();state['audit_marker']='changed';compact.atomic_json(self.ledger.path,state)
        with self.assertRaises(LedgerError):self.ledger.authorize_completion_window(self.cid,self.protocol,'auth',amendment,self.allowed)

    def test_closed_child_cannot_authorize_another_or_restart(self):
        amendment=self.amendment();self.authorize();entry=self.reserve();self.ledger.update(entry['attempt_id'],'submission_rejected')
        with self.assertRaises(DuplicateSubmission):self.reserve()
        with self.assertRaises(LedgerError):self.ledger.authorize_completion_window(self.cid,self.protocol,'auth',amendment,self.allowed)
        with self.assertRaises(DuplicateSubmission):self.ledger.authorize_completion_window('child2',self.protocol,'auth',self.amendment(),self.allowed)

    def test_boundary_stop_allows_current_batch_only(self):
        self.authorize();entry=self.reserve();self.ledger.request_stop(self.cid,'explicit user stop');self.finish(self.ledger,entry)
        self.assertEqual(self.ledger.window(self.cid)['outcome'],'operator_stop')
        self.assertFalse(self.ledger.window(self.cid)['backend_cancelled'])

    def test_completed_requires_all_whitelisted_operations(self):
        self.authorize();self.finish(self.ledger,self.reserve())
        with self.assertRaises(UnresolvedTask):self.ledger.close_main_window(self.cid,'completed','too early')

    def test_parent_tamper_detected_after_child_authorization(self):
        self.authorize();snapshot=self.ledger.snapshot();snapshot['execution_windows'][0]['reason']='changed';compact.atomic_json(self.ledger.path,snapshot)
        with self.assertRaises(LedgerError):self.ledger.snapshot()

    def test_whitelist_and_child_deadline_tampering_detected(self):
        self.authorize();original=self.ledger.snapshot()
        for field,value in [('deadline_epoch',self.clock.wall()+999),('policy',{**POLICY,'circuits':6000})]:
            state=copy.deepcopy(original);state['execution_windows'][-1][field]=value;compact.atomic_json(self.ledger.path,state)
            with self.assertRaises(LedgerError):self.ledger.snapshot()
        compact.atomic_json(self.ledger.path,original)

    def test_combined_attempt_limit_counts_parent(self):
        # Original parent already consumed 271 legal one-circuit attempts.
        state=copy.deepcopy(self.parent_state);template=state['entries'][0];state['entries']=[]
        for i in range(271):
            e=copy.deepcopy(template);e.update(attempt_id='p'+str(i),circuits=1,requested_shots=512,operation_id='old/'+str(i))
            specs=self.specs(1,'old'+str(i));e['metadata']=self.meta(specs,e['operation_id']);e['request_identity']=compact.request_identity(specs,512,'tianyan_sw','main');state['entries'].append(e)
        compact.atomic_json(self.ledger.path,state)
        with self.assertRaises(BudgetExceeded):self.amendment()

    def test_combined_circuit_limit_counts_parent(self):
        state=copy.deepcopy(self.parent_state);template=state['entries'][0];state['entries']=[]
        for i in range(265):
            n=20 if i<264 else 12;e=copy.deepcopy(template);e.update(attempt_id='p'+str(i),circuits=n,requested_shots=n*512,operation_id='old/'+str(i))
            specs=self.specs(n,'old'+str(i));e['metadata']=self.meta(specs,e['operation_id']);e['request_identity']=compact.request_identity(specs,512,'tianyan_sw','main');state['entries'].append(e)
        compact.atomic_json(self.ledger.path,state)
        with self.assertRaises(BudgetExceeded):self.amendment()

    def provider(self,mode=None):
        self.transport=MainTransport(self.ledger,self.clock)
        if mode:self.transport.mode=mode
        evidence=fixture_evidence(self.ledger,self.clock);evidence['capabilities']['approved_main_shots_limit']=512
        return CompletionAdapter(self.transport,self.ledger,evidence,sleep=self.clock.sleep,monotonic=self.clock.mono,
            execution_window_id=self.cid,execution_protocol_sha256=self.protocol)

    def adapter_fixture(self):
        raw=twelve_specs()[:4];evidence=fixture_evidence(self.ledger,self.clock)
        canonical=[circuit_spec(spec,evidence['capabilities']) for spec in raw]
        self.allowed[self.op]=self.allowed_value(canonical);self.authorize();return raw

    def test_incomplete_whitelist_cannot_authorize(self):
        self.allowed.pop(next(k for k in self.allowed if k!=self.op))
        with self.assertRaises(BudgetExceeded):self.authorize()

    def test_clock_rollback_before_activation_blocks_reservation(self):
        self.authorize();self.clock.advance(-1)
        with self.assertRaises(LedgerError):self.reserve()

    def test_adapter_raw_reads_and_live_check_use_new_authority(self):
        raw=self.adapter_fixture();provider=self.provider();out=provider.probability(raw,512,context={'operation_id':self.op})
        self.assertEqual(len(out),4);self.assertEqual(self.transport.submissions,1)
        self.assertEqual(self.ledger.snapshot()['entries'][-1]['state'],'validated')
        self.assertEqual(provider.execution_identity()['execution_window_kind'],WINDOW_KIND)

    def test_adapter_unknown_result_stops_no_retry(self):
        raw=self.adapter_fixture();provider=self.provider('empty')
        with self.assertRaises(UnknownSubmission):provider.probability(raw,512,context={'operation_id':self.op})
        queries=self.transport.queries
        with self.assertRaises(LedgerError):provider.probability(raw,512,context={'operation_id':self.op})
        self.assertEqual(self.transport.queries,queries);self.assertEqual(self.transport.submissions,1)


if __name__=='__main__':unittest.main()
