import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from ghost_policy import OnnxPolicy


def save_policy(path, bias, input_shape=(1, 156)):
    weights = np.zeros((156, 2), dtype=np.float32)
    weights[154, 0] = .5
    weights[153, 1] = .25
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['obs', 'weights'], ['linear']),
         helper.make_node('Add', ['linear', 'bias'], ['action'])],
        'policy', [helper.make_tensor_value_info('obs', TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(weights, 'weights'),
         numpy_helper.from_array(np.asarray(bias, dtype=np.float32), 'bias')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
    onnx.save(model, path)


def test_real_graph_uses_latest_motion_observation_and_clips_actions(tmp_path):
    path = tmp_path / 'policy.onnx'
    save_policy(path, [0., .3])
    policy = OnnxPolicy(path)
    obs = np.zeros((1, 156), dtype=np.float32)
    obs[0, 153:155] = [.4, -.8]
    np.testing.assert_allclose(policy.act(obs), [-.4, .4], atol=1e-6)
    save_policy(path, [2., -1.])
    np.testing.assert_array_equal(OnnxPolicy(path).act(obs), [1., 0.])


def test_nonfinite_model_action_is_rejected(tmp_path):
    path = tmp_path / 'bad.onnx'
    save_policy(path, [float('nan'), .3])
    with pytest.raises(ValueError, match='nonfinite'):
        OnnxPolicy(path).act(np.zeros((1, 156), dtype=np.float32))


def test_variable_input_shape_is_not_an_eligible_contract(tmp_path):
    path = tmp_path / 'variable.onnx'
    save_policy(path, [0., .3], input_shape=('batch', 156))
    with pytest.raises(ValueError, match='exactly one input'):
        OnnxPolicy(path)


def test_external_tensor_file_is_rejected_before_it_is_read(tmp_path):
    path = tmp_path / 'external.onnx'
    save_policy(path, [0., .3])
    model = onnx.load(path)
    tensor = model.graph.initializer[0]
    tensor.ClearField('raw_data')
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = 'location', 'missing-weights.bin'
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match='external tensor data'):
        OnnxPolicy(path)
