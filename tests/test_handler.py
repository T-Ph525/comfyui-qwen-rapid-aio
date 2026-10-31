from types import SimpleNamespace
import unittest
from rapidgen_handler import handler, install_history_guard
from tests.test_workflow_guard import example


class HandlerTests(unittest.TestCase):
    def job(self):
        workflow, images=example()
        return {'id':'test-job','input':{'workflow':workflow,'images':images}}

    def worker(self, status):
        uploads=[]
        upstream=SimpleNamespace(get_history=lambda prompt_id:{prompt_id:{'status':status,'outputs':{'6':{'images':[{'filename':'partial.png'}]}}}})
        def process(job):
            upstream.get_history('prompt')
            uploads.append('S3-upload')
            return {'images':[{'type':'s3_url','data':'https://example.invalid/image.png'}]}
        upstream.handler=process
        return install_history_guard(upstream),uploads

    def error(self, exception, node='RapidGenAgeGate'):
        return {'status_str':'error','completed':False,'messages':[['execution_error',{'node_type':node,'exception_type':exception}]]}

    def test_policy_rejection_stops_before_s3_and_keeps_charge(self):
        worker,uploads=self.worker(self.error('custom_nodes.rapidgen_age_gate.policy.PolicyRejected'))
        result=handler(self.job(),worker)
        self.assertEqual(result['billing_outcome'],'CHARGE')
        self.assertEqual(result['error_code'],'POLICY_REJECTED')
        self.assertEqual(uploads,[])
        self.assertNotIn('images',result)

    def test_scanner_fault_stops_before_s3_and_marks_server_failure(self):
        worker,uploads=self.worker(self.error('ScannerUnavailable'))
        result=handler(self.job(),worker)
        self.assertEqual(result['billing_outcome'],'REFUND_SERVER_FAILURE')
        self.assertEqual(uploads,[])

    def test_other_errors_never_return_partial_images_or_auto_refund(self):
        worker,uploads=self.worker(self.error('RuntimeError','KSampler'))
        result=handler(self.job(),worker)
        self.assertEqual(result['billing_outcome'],'REVIEW')
        self.assertEqual(uploads,[])

    def test_success_preserves_s3_output(self):
        worker,uploads=self.worker({'status_str':'success','completed':True,'messages':[]})
        result=handler(self.job(),worker)
        self.assertEqual(result['scanner'],'face-age-v1')
        self.assertEqual(result['scanner_model'],'mivolo-v2-age-v1')
        self.assertEqual(len(result['images']),1)
        self.assertEqual(uploads,['S3-upload'])

    def test_missing_gate_cannot_bypass_wrapper(self):
        worker,uploads=self.worker({'status_str':'success','completed':True,'messages':[]})
        job=self.job()
        job['input']['workflow']['99']={'class_type':'UnsafeUploader','inputs':{}}
        self.assertEqual(handler(job,worker)['error_code'],'INPUT_REJECTED')
        self.assertEqual(uploads,[])


class TextToImageHandlerTests(HandlerTests):
    def job(self):
        from tests.test_workflow_guard import text_example
        workflow, images = text_example()
        return {'id': 'text-job', 'input': {'mode': 'text-to-image', 'workflow': workflow, 'images': images}}
