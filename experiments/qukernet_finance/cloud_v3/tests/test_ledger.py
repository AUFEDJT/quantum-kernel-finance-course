"""Offline authority, quota, deadline, stop, and reviewed-adapter integration."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
sys.path.insert(0,str(CODE/'cloud_v2/tests'))
from cloud_v2 import ledger as old
from cloud_v2.cloud import UnknownSubmission,InvalidResult
from cloud_v2.preflight import PreflightBlocked
from test_main_window import MainTransport
from test_authorized_window_integration import VirtualClock,twelve_specs,fixture_evidence
from cloud_v3.ledger import (PersistentLedger,TianyanCloudAdapter,MAIN_POLICY,WINDOW_KIND,
                            LedgerError,BudgetExceeded,UnresolvedTask,DuplicateSubmission)


class CompactLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.clock=VirtualClock();self.scope='offline-v3-ledger'
        prior=old.PersistentLedger(self.temp.name,self.scope,clock=self.clock.wall)
        e=prior.reserve(1,128,'preflight','historical-probe',{})
        prior.update(e['attempt_id'],'submitted',job_ids=['historical-job'])
        prior.update(e['attempt_id'],'validated')
        prior.authorize_main_window('old-main','a'*64,'historical authorized fixture')
        specs=self.specs(2,'old')
        e=prior.reserve(2,512,'main','old-main-fingerprint',self.meta(specs,'old-op','a'*64),window_id='old-main')
        prior.update(e['attempt_id'],'submitted',job_ids=['old-main-1','old-main-2'])
        prior.update(e['attempt_id'],'validated');prior.close_main_window('old-main','completed','fixture')
        self.before=prior.snapshot();self.clock.advance(72*3600+10)
        self.ledger=PersistentLedger(self.temp.name,self.scope,clock=self.clock.wall)
        self.wid='compact-pilot';self.protocol='c'*64
        self.ledger.authorize_main_window(self.wid,self.protocol,'explicit offline fixture')
        self.serial=0

    def tearDown(self):self.temp.cleanup()

    @staticmethod
    def specs(n,prefix='x'):
        return [{'id':f'{prefix}-{i}','sha256':hashlib.sha256(f'{prefix}-{i}'.encode()).hexdigest(),'qubits':8} for i in range(n)]

    @staticmethod
    def meta(specs,op,protocol='c'*64):
        return {'circuits':specs,'backend':'tianyan_sw','protocol_sha256':protocol,'context':{'operation_id':op}}

    def reserve(self,n=4,op=None):
        self.serial+=1;op=op or f'phase/{self.serial}'
        return self.ledger.reserve(n,512,'main',f'fp/{self.serial}',self.meta(self.specs(n,op),op),window_id=self.wid)

    def finish(self,e):
        self.ledger.update(e['attempt_id'],'submitted',job_ids=[e['attempt_id']+f'-{i}' for i in range(e['circuits'])])
        self.ledger.update(e['attempt_id'],'validated')

    def provider(self,mode=None,**kwargs):
        self.transport=MainTransport(self.ledger,self.clock)
        if mode:self.transport.mode=mode
        proof=fixture_evidence(self.ledger,self.clock);proof['capabilities']['approved_main_shots_limit']=512
        return TianyanCloudAdapter(self.transport,self.ledger,proof,sleep=self.clock.sleep,monotonic=self.clock.mono,
            execution_window_id=self.wid,execution_protocol_sha256=kwargs.pop('protocol',self.protocol),**kwargs)

    def test_legacy_clock_entries_windows_preserved(self):
        self.finish(self.reserve())
        after=self.ledger.snapshot()
        for k in ('schema_version','limits','started_epoch','deadline_epoch'):
            self.assertEqual(after[k],self.before[k])
        self.assertEqual(after['entries'][:2],self.before['entries'])
        self.assertEqual(after['execution_windows'][0],self.before['execution_windows'][0])
        self.assertEqual(self.ledger.window(self.wid)['kind'],WINDOW_KIND)
        self.assertEqual(MAIN_POLICY['circuits'],5292)
        self.assertEqual(MAIN_POLICY['shots'],2709504)

    def test_first_probe_reservation_starts_three_hours_not_authorization(self):
        self.assertIsNone(self.ledger.window(self.wid)['deadline_epoch'])
        self.clock.advance(200);start=self.clock.wall();e=self.reserve(4)
        w=self.ledger.window(self.wid)
        self.assertEqual(w['started_epoch'],start);self.assertEqual(w['deadline_epoch'],start+10800)
        self.assertEqual(e['observation_deadline_epoch'],start+300)

    def test_restart_keeps_deadline(self):
        self.finish(self.reserve());deadline=self.ledger.effective_deadline(self.wid)
        self.clock.advance(100)
        again=PersistentLedger(self.temp.name,self.scope,clock=self.clock.wall)
        again.authorize_main_window(self.wid,self.protocol,'explicit offline fixture')
        self.assertEqual(again.effective_deadline(self.wid),deadline)

    def test_last_batch_deadline_is_capped_by_global_deadline(self):
        self.finish(self.reserve());end=self.ledger.effective_deadline(self.wid)
        self.clock.advance(end-self.clock.wall()-50);e=self.reserve()
        self.assertEqual(e['observation_deadline_epoch'],end)
        self.ledger.update(e['attempt_id'],'submitted',job_ids=['last-1','last-2','last-3','last-4'])
        self.clock.advance(50)
        with self.assertRaises(BudgetExceeded):self.ledger.update(e['attempt_id'],'validated')

    def test_new_reservation_after_three_hours_rejected(self):
        self.finish(self.reserve());self.clock.advance(10800)
        with self.assertRaises(BudgetExceeded):self.reserve()

    def test_late_validation_after_batch300_rejected(self):
        e=self.reserve();self.ledger.update(e['attempt_id'],'submitted',job_ids=['late'])
        self.clock.advance(300)
        with self.assertRaises(BudgetExceeded):self.ledger.update(e['attempt_id'],'validated')

    def test_duplicate_operation_even_new_window_and_changed_content(self):
        self.finish(self.reserve(op='same-operation'))
        self.ledger.close_main_window(self.wid,'completed','fixture complete')
        self.wid='second';self.ledger.authorize_main_window(self.wid,self.protocol,'new explicit fixture')
        with self.assertRaises(DuplicateSubmission):self.reserve(3,'same-operation')

    def test_independent_method_operations_can_use_same_circuits(self):
        specs=self.specs(4,'same-physical')
        for method in ('q','random'):
            e=self.ledger.reserve(4,512,'main','same-'+method,self.meta(specs,method),window_id=self.wid)
            self.finish(e)
        self.assertEqual(len(self.ledger.snapshot()['entries']),4)

    def test_unknown_reservation_stops_and_blocks_fresh_authority(self):
        e=self.reserve();self.ledger.update(e['attempt_id'],'submission_unknown')
        self.assertEqual(self.ledger.window(self.wid)['state'],'closed')
        with self.assertRaises(DuplicateSubmission):self.reserve()
        with self.assertRaises(UnresolvedTask):self.ledger.authorize_main_window('new',self.protocol,'not a bypass')

    def test_historical_unresolved_attempt_cannot_be_bypassed(self):
        self.ledger.close_main_window(self.wid,'client_stopped','unstarted fixture')
        state=self.ledger.snapshot();state['entries'][0]['state']='observation_unknown'
        old.atomic_json(self.ledger.path,state)
        with self.assertRaises(UnresolvedTask):self.ledger.authorize_main_window('new',self.protocol,'no bypass')

    def test_active_old_window_blocks_new_authority(self):
        self.ledger.close_main_window(self.wid,'client_stopped','fixture')
        state=self.ledger.snapshot();state['execution_windows'][0]['state']='running'
        old.atomic_json(self.ledger.path,state)
        with self.assertRaises(UnresolvedTask):self.ledger.authorize_main_window('new',self.protocol,'no bypass')

    def test_stop_without_inflight_closes_and_persists(self):
        self.ledger.request_stop(self.wid,'user selected stop')
        again=PersistentLedger(self.temp.name,self.scope,clock=self.clock.wall)
        self.assertTrue(again.window(self.wid)['stop_requested'])
        with self.assertRaises(DuplicateSubmission):again.effective_deadline(self.wid)
        with self.assertRaises(DuplicateSubmission):self.reserve()

    def test_boundary_stop_allows_current_result_only(self):
        e=self.reserve();self.ledger.request_stop(self.wid,'after current batch')
        self.assertEqual(self.ledger.window(self.wid)['state'],'running')
        with self.assertRaises(DuplicateSubmission):self.reserve()
        self.finish(e)
        self.assertEqual(self.ledger.window(self.wid)['outcome'],'operator_stop')
        self.assertFalse(self.ledger.window(self.wid)['backend_cancelled'])

    def test_unstarted_or_unresolved_cannot_mark_completed(self):
        with self.assertRaises(UnresolvedTask):self.ledger.close_main_window(self.wid,'completed','not complete')
        self.reserve()
        with self.assertRaises(UnresolvedTask):self.ledger.close_main_window(self.wid,'completed','not complete')

    def test_271_attempt_cap_independent_of_circuit_cap(self):
        # Validated tiny batches hit the attempt cap before the circuit cap.
        for _ in range(271):self.finish(self.reserve(1))
        with self.assertRaises(BudgetExceeded):self.reserve(1)
        self.assertEqual(len(self.ledger.snapshot()['entries'])-2,271)

    def test_5292_circuit_and_shot_cap(self):
        for _ in range(264):self.finish(self.reserve(20))
        self.finish(self.reserve(12))
        with self.assertRaises(BudgetExceeded):self.reserve(1)
        entries=self.ledger.snapshot()['entries'][2:]
        self.assertEqual(sum(e['circuits'] for e in entries),5292)
        self.assertEqual(sum(e['requested_shots'] for e in entries),2709504)

    def test_rejected_attempt_consumes_shots_and_no_new_submit(self):
        e=self.reserve();self.ledger.update(e['attempt_id'],'submission_rejected')
        self.assertEqual(self.ledger.snapshot()['entries'][-1]['requested_shots'],2048)
        with self.assertRaises(DuplicateSubmission):self.reserve()

    def test_invalid_request_parameters_rejected(self):
        for n,shots,phase,qubits in ((21,512,'main',8),(4,128,'main',8),(4,512,'preflight',8),(4,512,'main',4)):
            specs=self.specs(n);[s.update(qubits=qubits) for s in specs]
            with self.subTest(n=n,shots=shots,phase=phase,qubits=qubits):
                with self.assertRaises(BudgetExceeded):self.ledger.reserve(n,shots,phase,'invalid',self.meta(specs,'invalid'),window_id=self.wid)

    def test_no_unwindowed_or_legacy_authorization_bypass(self):
        with self.assertRaises(LedgerError):self.ledger.reserve(4,512,'main','raw',self.meta(self.specs(4),'raw'))
        with self.assertRaises(LedgerError):self.ledger.authorize_window('legacy','a'*64,'b'*64,'not permitted')

    def test_frozen_window_policy_or_deadline_tamper_fails(self):
        self.finish(self.reserve());original=self.ledger.snapshot()
        for key,value in (('deadline_epoch',self.clock.wall()+50000),('policy',{**MAIN_POLICY,'seconds':7200})):
            bad=copy.deepcopy(original);bad['execution_windows'][-1][key]=value;old.atomic_json(self.ledger.path,bad)
            with self.assertRaises(LedgerError):self.ledger.snapshot()
            old.atomic_json(self.ledger.path,original)

    def test_old_window_policy_tamper_still_fails(self):
        state=self.ledger.snapshot();state['execution_windows'][0]['policy']['circuits']+=1
        old.atomic_json(self.ledger.path,state)
        with self.assertRaises(LedgerError):self.ledger.snapshot()

    def test_old_authorized_preflight_window_remains_valid(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=old.PersistentLedger(root,'old-preflight-account',clock=self.clock.wall)
            specs=self.specs(12,'legacy-probe')
            fingerprint=old.request_identity(specs,128,'tianyan_sw','preflight')
            ledger.authorize_window('old-probe','a'*64,fingerprint,'old explicit fixture')
            e=ledger.reserve(12,128,'preflight','legacy-fingerprint',
                {'backend':'tianyan_sw','circuits':specs},window_id='old-probe')
            ledger.update(e['attempt_id'],'submitted',job_ids=[f'legacy-{i}' for i in range(12)])
            ledger.update(e['attempt_id'],'validated');before=ledger.snapshot()
            compact=PersistentLedger(root,'old-preflight-account',clock=self.clock.wall)
            compact.authorize_main_window('compact',self.protocol,'new explicit fixture')
            self.assertEqual(compact.snapshot()['execution_windows'][0],before['execution_windows'][0])
            self.assertEqual(compact.snapshot()['entries'],before['entries'])

    def test_adapter_validates_raw_reads_and_new_kind(self):
        provider=self.provider();out=provider.probability(twelve_specs()[:4],512,context={'operation_id':'analytic-preflight'})
        self.assertEqual(len(out),4);self.assertEqual(self.transport.submissions,1)
        self.assertEqual(provider.execution_identity()['execution_window_kind'],WINDOW_KIND)
        self.assertEqual(self.ledger.snapshot()['entries'][-1]['state'],'validated')

    def test_adapter_wrong_protocol_fails_before_any_submission(self):
        with self.assertRaises(PreflightBlocked):self.provider(protocol='f'*64)
        self.assertEqual(self.transport.submissions,0)

    def test_adapter_empty_results_stops_no_automatic_requery(self):
        provider=self.provider('empty')
        with self.assertRaises(UnknownSubmission):provider.probability(twelve_specs()[:4],512,context={'operation_id':'empty'})
        queries=self.transport.queries
        with self.assertRaises(LedgerError):provider.probability(twelve_specs()[:4],512,context={'operation_id':'next'})
        self.assertEqual(self.transport.queries,queries);self.assertEqual(self.transport.submissions,1)
        self.assertEqual(self.ledger.window(self.wid)['state'],'closed')

    def test_adapter_observation_stops_at_remaining_global_time(self):
        provider=self.provider()
        provider.probability(twelve_specs()[:4],512,context={'operation_id':'first'})
        deadline=self.ledger.effective_deadline(self.wid)
        self.clock.advance(deadline-self.clock.wall()-25)
        self.transport.mode='empty';before=self.clock.wall()
        with self.assertRaises(UnknownSubmission):
            provider.probability(twelve_specs()[:4],512,context={'operation_id':'near-limit'})
        self.assertLessEqual(self.clock.wall()-before,25)
        self.assertEqual(self.ledger.snapshot()['entries'][-1]['observation_deadline_epoch'],deadline)

    def test_adapter_boundary_stop_finishes_current_batch_and_never_queries_next(self):
        provider=self.provider();query=self.transport.query
        def stop_during_query(ids,deadline):
            self.ledger.request_stop(self.wid,'finish only current batch')
            return query(ids,deadline)
        self.transport.query=stop_during_query
        out=provider.probability(twelve_specs()[:4],512,context={'operation_id':'current'})
        self.assertEqual(len(out),4)
        self.assertEqual(self.ledger.window(self.wid)['outcome'],'operator_stop')
        queries=self.transport.queries
        with self.assertRaises(LedgerError):
            provider.probability(twelve_specs()[:4],512,context={'operation_id':'next'})
        self.assertEqual(self.transport.queries,queries)
        self.assertEqual(self.transport.submissions,1)

    def test_adapter_submission_timeout_does_not_query_or_retry(self):
        provider=self.provider('submit_timeout')
        with self.assertRaises(UnknownSubmission):provider.probability(twelve_specs()[:4],512,context={'operation_id':'timeout'})
        self.assertEqual(self.transport.submissions,1);self.assertEqual(self.transport.queries,0)
        self.assertEqual(self.ledger.snapshot()['entries'][-1]['state'],'submission_unknown')

    def test_adapter_frozen_statistical_validator_failure_stops(self):
        def invalid(results):raise ValueError('frozen acceptance fixture')
        provider=self.provider(result_validator=invalid)
        with self.assertRaises(InvalidResult):provider.probability(twelve_specs()[:4],512,context={'operation_id':'invalid'})
        self.assertEqual(self.ledger.snapshot()['entries'][-1]['state'],'received_invalid')


if __name__=='__main__':unittest.main()
