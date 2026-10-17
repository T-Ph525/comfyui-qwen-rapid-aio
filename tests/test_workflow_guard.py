from copy import deepcopy
import json
from pathlib import Path
import unittest

from workflow_guard import InputRejected, protect_workflow

ROOT = Path(__file__).parents[1]


def example():
    graph = json.loads((ROOT / 'api-workflow.json').read_text())
    return graph, [{'name': 'images.png', 'image': 'base64-placeholder'}]


class WorkflowTests(unittest.TestCase):
    def test_existing_example_is_rewired_without_mutating_input(self):
        graph, images = example()
        original = deepcopy(graph)
        protected = protect_workflow(graph, images)
        self.assertEqual(graph, original)
        pre, slot = protected['10']['inputs']['prompt']
        self.assertEqual(slot, 4)
        self.assertEqual(protected[pre]['class_type'], 'RapidGenAgeGate')
        post, slot = protected['6']['inputs']['images']
        self.assertEqual(slot, 0)
        self.assertEqual(protected[post]['class_type'], 'RapidGenAgeGate')
        self.assertEqual(protected[post]['inputs']['image1'], ['5', 0])
        self.assertEqual(protected[pre]['inputs']['image1'], ['8', 0])

    def test_ungated_legacy_workflow_is_protected(self):
        graph, images = example()
        del graph['20'], graph['21']
        graph['10']['inputs'].update({'prompt':'Restyle the outfit', 'image1':['8',0]})
        graph['6']['inputs']['images']=['5',0]
        protected=protect_workflow(graph, images)
        self.assertEqual(sum(node['class_type']=='RapidGenAgeGate' for node in protected.values()),2)

    def test_every_reference_goes_through_the_input_gate(self):
        graph, images = example()
        graph['11']={'class_type':'LoadImage','inputs':{'image':'reference.png'}}
        graph['10']['inputs']['image2']=['11',0]
        images.append({'name':'reference.png','image':'placeholder'})
        protected=protect_workflow(graph,images)
        pre=protected['10']['inputs']['prompt'][0]
        self.assertEqual(protected[pre]['inputs']['image2'],['11',0])
        self.assertEqual(protected['10']['inputs']['image2'],[pre,1])

    def test_extra_preview_cannot_export_unchecked_images(self):
        graph, images = example()
        graph['99']={'class_type':'PreviewImage','inputs':{'images':['8',0]}}
        protected=protect_workflow(graph,images)
        gate=protected['99']['inputs']['images'][0]
        self.assertEqual(protected[gate]['class_type'],'RapidGenAgeGate')

    def test_unknown_output_node_is_rejected(self):
        graph, images = example()
        graph['99']={'class_type':'S3Upload','inputs':{'images':['5',0]}}
        with self.assertRaises(InputRejected):protect_workflow(graph,images)

    def test_missing_upload_and_path_traversal_are_rejected(self):
        graph, images = example()
        for invalid in [[],[{'name':'../secret.png','image':'x'}]]:
            with self.assertRaises(InputRejected):protect_workflow(graph,invalid)

    def test_cycle_is_rejected_before_queueing(self):
        graph, images=example()
        graph['5']['inputs']['samples']=['5',0]
        with self.assertRaises(InputRejected):protect_workflow(graph,images)

    def test_ui_workflow_links_are_consistent(self):
        workflow=json.loads((ROOT/'workflow.json').read_text())
        nodes={node['id']:node for node in workflow['nodes']}
        for link_id, source, source_slot, target, target_slot, kind in workflow['links']:
            self.assertIn(link_id,nodes[source]['outputs'][source_slot]['links'])
            self.assertEqual(nodes[target]['inputs'][target_slot]['link'],link_id)
        self.assertEqual(nodes[6]['type'],'SaveImage')
        self.assertEqual(nodes[1]['widgets_values'][0], 'v23/Qwen-Rapid-AIO-NSFW-v23.safetensors')
