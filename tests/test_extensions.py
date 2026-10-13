import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from harness import edge_profile
from harness.whisper import word_errors
from serving.benchmark import benchmark_level, parse_metrics, request
from serving.run import kv_capacity, run_name, server_command, trtllm_config_text
from serving.trtllm_server import collect_iteration_stats


def serving_args(**kwargs):
    values = dict(backend='vllm', quantization='none', tensor_parallel_size=1,
                  kv_capacity_tokens=65536, concurrency=[1, 4], port=8000)
    return SimpleNamespace(**(values | kwargs))


def test_fp8_and_tp2_preserve_sharded_kv_payload_budget():
    single = serving_args()
    dual = serving_args(quantization='fp8', tensor_parallel_size=2)
    command = server_command(dual)
    assert command[command.index('--quantization') + 1] == 'fp8'
    assert command[command.index('--tensor-parallel-size') + 1] == '2'
    assert int(command[command.index('--kv-cache-memory-bytes') + 1]) * 2 == kv_capacity(single)
    assert run_name(single) != run_name(dual)


def test_trtllm_launch_uses_config_alias_available_in_cuda12_release():
    args = serving_args(backend='tensorrt-llm', trtllm_config='/tmp/pinned.yaml')
    command = server_command(args)
    assert 'serving.trtllm_server' in command
    assert command[command.index('--extra_llm_api_options') + 1] == '/tmp/pinned.yaml'
    assert command[command.index('--backend') + 1] == 'pytorch'
    # Revision lives in the YAML for compatibility with TRT-LLM 1.0's CLI.
    assert '--revision' not in command
    from serving.benchmark import MODEL_REVISION
    config = trtllm_config_text(args)
    assert f'revision: {MODEL_REVISION}\n' in config
    assert f'tokenizer_revision: {MODEL_REVISION}\n' in config
    assert 'enable_block_reuse: false' in config


def test_trtllm_json_metrics_retain_real_block_counts():
    stats = [{'iter': 3, 'numActiveRequests': 1, 'kvCacheStats': {
        'usedNumBlocks': 4, 'maxNumBlocks': 16, 'tokensPerBlock': 32}},
        {'iter': 4, 'numActiveRequests': 0, 'kvCacheStats': {'usedNumBlocks': 0, 'maxNumBlocks': 16}}]
    samples = parse_metrics(json.dumps(stats), 'tensorrt-llm')
    assert samples[0]['fraction'] == .25
    assert samples[0]['source_active'] and not samples[1]['source_active']
    assert samples[0]['used_blocks'] == 4
    assert parse_metrics('[]', 'tensorrt-llm') == []
    with pytest.raises(ValueError):
        parse_metrics('{}', 'tensorrt-llm')


def test_reject_short_decode_in_fixed_length_workload():
    body = ('data: {"choices":[{"text":"x"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":16,"completion_tokens":1}}\n\n'
            'data: [DONE]\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))) as client:
            return await request(client, 'http://test', 'model', 0, 16, 8, 7)
    result = asyncio.run(run())
    assert 'fixed-length' in result.error


def test_word_error_rate_counts_insertions_and_case():
    assert word_errors('HELLO, world!', 'hello world') == (0, 2)
    assert word_errors('the quick brown fox', 'the slow fox jumped') == (3, 4)
    assert word_errors('hello', 'hello extra extra') == (2, 1)


@pytest.mark.parametrize('refused,crash', [(False, False), (True, False), (False, True)])
def test_edge_cap_restored_and_refusal_labeled(tmp_path, monkeypatch, refused, crash):
    args = SimpleNamespace(out_dir=str(tmp_path), gpu='0', power_limit=200)
    current = 450
    calls = []
    def query(gpu):
        return dict(uuid='GPU-test', current_watts=current, min_watts=100, max_watts=450)
    def smi(gpu, *arguments):
        nonlocal current
        calls.append(arguments)
        value = float(arguments[-1])
        if refused and value == 200:
            return dict(returncode=1, stdout='', stderr='permission denied')
        current = value
        return dict(returncode=0, stdout='', stderr='')
    def workload(args, root, name, uuid):
        if crash and name == 'capped':
            raise RuntimeError('workload failed')
        return dict(returncode=0)
    monkeypatch.setattr(edge_profile, 'query_limit', query)
    monkeypatch.setattr(edge_profile, 'smi', smi)
    monkeypatch.setattr(edge_profile, 'workload', workload)
    if crash:
        with pytest.raises(RuntimeError, match='workload failed'):
            edge_profile.run_profile(args)
    else:
        assert edge_profile.run_profile(args) == 0
    report = json.loads((tmp_path / 'profile.json').read_text())
    assert report['label'] == 'edge-style, not Jetson'
    assert current == 450 and report['restore_verified']
    if refused:
        assert report['cap_status'] == 'refused_or_unverified'
        assert 'cap-refused-uncapped' in report['runs']
    assert calls[-1] == ('-pl', '450') or calls[-1] == ('-pl', '450.0')


def test_whisper_fixed_mask_export_keeps_causal_semantics_and_dynamic_position(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    ort = pytest.importorskip('onnxruntime')
    from harness.whisper import components
    from training.export_tensorrt import export_model
    config = transformers.WhisperConfig(vocab_size=64, d_model=16, encoder_layers=1,
        decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_target_positions=32,
        max_source_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        decoder_start_token_id=1, suppress_tokens=[], begin_suppress_tokens=[])
    config._attn_implementation = 'eager'
    torch.manual_seed(7)
    model = transformers.WhisperForConditionalGeneration(config).eval()
    _, decoder = components(model, 8)
    ids = torch.randint(3, 64, (2, 8))
    hidden = torch.randn(2, 7, 16)
    position = torch.tensor([3])
    path = tmp_path / 'decoder.onnx'
    export_model(decoder, (ids, hidden, position), path,
                 ('input_ids', 'hidden', 'position'), ('logits',))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        reference = model.proj_out(model.model.decoder(input_ids=ids,
            encoder_hidden_states=hidden, use_cache=False, return_dict=False)[0])
        for index in (3, 6):
            position.fill_(index)
            actual = decoder(ids, hidden, position)
            torch.testing.assert_close(actual, reference[:, index], atol=1e-6, rtol=1e-5)
            exported = session.run(None, dict(input_ids=ids.numpy(), hidden=hidden.numpy(), position=position.numpy()))[0]
            torch.testing.assert_close(torch.from_numpy(exported), actual, atol=1e-6, rtol=1e-5)


def test_trtllm_excludes_warmup_stats_drained_before_load():
    calls = 0
    def handler(req):
        nonlocal calls
        if req.url.path == '/metrics':
            calls += 1
            fraction = .9 if calls == 1 else .25
            return httpx.Response(200, json=[{'iter': calls, 'numActiveRequests': 1,
                'kvCacheStats': {'usedNumBlocks': fraction * 100, 'maxNumBlocks': 100}}])
        body = ('data: {"choices":[{"text":"x"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":16,"completion_tokens":2}}\n\n'
                'data: [DONE]\n\n')
        return httpx.Response(200, text=body)
    args = SimpleNamespace(requests=2, base_url='http://test', metrics_url=None,
        backend='tensorrt-llm', metrics_interval=.001, model='model', input_tokens=16,
        output_tokens=2, seed=7, kv_capacity_bytes=1024)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await benchmark_level(client, args, 1)
    row = asyncio.run(run())
    assert row['successful'] == 2
    assert row['kv_cache']['peak_sampled_usage_fraction'] == .25
    assert row['kv_cache']['samples'][0]['elapsed_seconds'] < 0


def test_trtllm_metrics_preserve_stats_before_empty_deque_error():
    class LLM:
        async def get_stats_async(self, timeout):
            yield {'iter': 1, 'kvCacheStats': {'usedNumBlocks': 3, 'maxNumBlocks': 16}}
            raise IndexError('pop from an empty deque')
    assert asyncio.run(collect_iteration_stats(LLM())) == [
        {'iter': 1, 'kvCacheStats': {'usedNumBlocks': 3, 'maxNumBlocks': 16}}]


def test_trtllm_metrics_return_collected_stats_when_producer_stays_active():
    class LLM:
        async def get_stats_async(self, timeout):
            yield {'iter': 2}
            await asyncio.sleep(1)
    assert asyncio.run(collect_iteration_stats(LLM(), timeout=.001)) == [{'iter': 2}]
