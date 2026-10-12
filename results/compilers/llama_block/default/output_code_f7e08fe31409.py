
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /workspace/gpusuite/results/compilers/llama_block/default/cache/zd/czd2mcdqb6oedjegh7vn3n75ayuv65rwa53dkbkp3xiwxwp4h3mc.py
# Source Nodes: [add, h, mean, mul, pow_1, rsqrt, to, xf], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# h => mul_1
# mean => mean
# mul => mul
# pow_1 => pow_1
# rsqrt => rsqrt
# to => convert_element_type_1
# xf => convert_element_type
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=128), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '7386E2E77C6CAE88D1A4A9C54EFA2E23CD53BB2F7ACF1FCAE0B7664C6BD05A22', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 4096.0
        tmp9 = tmp4 / tmp8
        tmp10 = 1e-05
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr1 + (r1 + (4096*x0)), tmp16, rmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /workspace/gpusuite/results/compilers/llama_block/default/cache/n6/cn6pyo6ipm3b56kv4c7kaw5hj5hsw73tuzguzmuvsm2ri7camqil.py
# Source Nodes: [a, cat, cat_1, k_1, mul_2, mul_3, mul_4, mul_5, q_1], Original ATen: [aten._scaled_dot_product_flash_attention, aten.add, aten.cat, aten.mul]
# a => _scaled_dot_product_flash_attention
# cat => cat
# cat_1 => cat_1
# k_1 => add_2
# mul_2 => mul_2
# mul_3 => mul_3
# mul_4 => mul_4
# mul_5 => mul_5
# q_1 => add_1
triton_poi_fused__scaled_dot_product_flash_attention_add_cat_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=128), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_add_cat_mul_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '7386E2E77C6CAE88D1A4A9C54EFA2E23CD53BB2F7ACF1FCAE0B7664C6BD05A22', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (x0 + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 64, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tl.load(in_ptr0 + (64 + x3), tmp7, other=0.0).to(tl.float32)
    tmp9 = -tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tmp12 = tmp3 >= tmp6
    tmp13 = tl.full([1], 128, tl.int64)
    tmp14 = tmp3 < tmp13
    tmp15 = tl.load(in_ptr0 + ((-64) + x3), tmp12, other=0.0).to(tl.float32)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp7, tmp11, tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp2 + tmp20
    tmp23 = tmp22 * tmp1
    tmp24 = tl.load(in_ptr3 + (64 + x3), tmp7, other=0.0).to(tl.float32)
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp7, tmp25, tmp26)
    tmp28 = tl.load(in_ptr3 + ((-64) + x3), tmp12, other=0.0).to(tl.float32)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp12, tmp28, tmp29)
    tmp31 = tl.where(tmp7, tmp27, tmp30)
    tmp32 = tmp31 * tmp19
    tmp33 = tmp23 + tmp32
    tl.store(out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: /workspace/gpusuite/results/compilers/llama_block/default/cache/fd/cfdei2uxeeek3b6a3xdtbcoim4zwqnjqrjqn5mlploor5kg7sd26.py
# Source Nodes: [add_4, h_1, mean_1, mul_6, pow_2, rsqrt_1, to_1, x, xf_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_4 => add_4
# h_1 => mul_7
# mean_1 => mean_1
# mul_6 => mul_6
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# to_1 => convert_element_type_11
# x => add_3
# xf_1 => convert_element_type_10
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=128), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '7386E2E77C6CAE88D1A4A9C54EFA2E23CD53BB2F7ACF1FCAE0B7664C6BD05A22', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r1 + (4096*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 4096.0
        tmp13 = tmp6 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r1 + (4096*x0)), tmp20, rmask)
''', device_str='cuda')


# kernel path: /workspace/gpusuite/results/compilers/llama_block/default/cache/2c/c2c52iaz2gyiz7cqq4rkylhcedefj7uoq2jenrpky4dwmbpbdkiy.py
# Source Nodes: [mul_8, silu], Original ATen: [aten.mul, aten.silu]
# mul_8 => mul_9
# silu => convert_element_type_14, convert_element_type_15, mul_8, sigmoid
triton_poi_fused_mul_silu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=128), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '7386E2E77C6CAE88D1A4A9C54EFA2E23CD53BB2F7ACF1FCAE0B7664C6BD05A22', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22544384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /workspace/gpusuite/results/compilers/llama_block/default/cache/4i/c4ixwthtano4iwst22z6hp5ict6s2phkk23bkjhte5ont26ilbsq.py
# Source Nodes: [add_5, x], Original ATen: [aten.add]
# add_5 => add_5
# x => add_3
triton_poi_fused_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=128), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '7386E2E77C6CAE88D1A4A9C54EFA2E23CD53BB2F7ACF1FCAE0B7664C6BD05A22', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2048, 4096), (8388608, 4096, 1))
    assert_size_stride(arg1_1, (4096, ), (1, ))
    assert_size_stride(arg2_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg3_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg4_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg5_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg6_1, (1, 1, 2048, 128), (262144, 262144, 128, 1))
    assert_size_stride(arg7_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg8_1, (4096, ), (1, ))
    assert_size_stride(arg9_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg10_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg11_1, (4096, 11008), (11008, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 2048, 4096), (8388608, 4096, 1), torch.float16)
        # Source Nodes: [add, h, mean, mul, pow_1, rsqrt, to, xf], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, buf1, 2048, 4096, grid=grid(2048), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg2_1, (4096, 4096), (1, 4096), 0), out=buf2)
        del arg2_1
        buf3 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg3_1, (4096, 4096), (1, 4096), 0), out=buf3)
        del arg3_1
        buf4 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float16)
        # Source Nodes: [matmul_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg4_1, (4096, 4096), (1, 4096), 0), out=buf4)
        del arg4_1
        buf5 = reinterpret_tensor(buf1, (1, 32, 2048, 128), (8388608, 128, 4096, 1), 0); del buf1  # reuse
        buf6 = empty_strided_cuda((1, 32, 2048, 128), (8388608, 128, 4096, 1), torch.float16)
        # Source Nodes: [a, cat, cat_1, k_1, mul_2, mul_3, mul_4, mul_5, q_1], Original ATen: [aten._scaled_dot_product_flash_attention, aten.add, aten.cat, aten.mul]
        triton_poi_fused__scaled_dot_product_flash_attention_add_cat_mul_1.run(buf2, arg5_1, arg6_1, buf3, buf5, buf6, 8388608, grid=grid(8388608), stream=stream0)
        del arg5_1
        del arg6_1
        del buf2
        del buf3
        # Source Nodes: [a, cat, cat_1, k_1, mul_2, mul_3, mul_4, mul_5, q_1], Original ATen: [aten._scaled_dot_product_flash_attention, aten.add, aten.cat, aten.mul]
        buf7 = aten._scaled_dot_product_flash_attention.default(buf5, buf6, reinterpret_tensor(buf4, (1, 32, 2048, 128), (8388608, 128, 4096, 1), 0), 0.0, True, scale=0.08838834764831843)
        del buf4
        buf8 = buf7[0]
        del buf7
        buf13 = reinterpret_tensor(buf6, (2048, 4096), (4096, 1), 0); del buf6  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 4096), (1, 4096), 0), out=buf13)
        del arg7_1
        buf15 = reinterpret_tensor(buf8, (1, 2048, 4096), (8388608, 4096, 1), 0); del buf8  # reuse
        # Source Nodes: [add_4, h_1, mean_1, mul_6, pow_2, rsqrt_1, to_1, x, xf_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2.run(arg0_1, buf13, arg8_1, buf15, 2048, 4096, grid=grid(2048), stream=stream0)
        del arg8_1
        buf16 = empty_strided_cuda((2048, 11008), (11008, 1), torch.float16)
        # Source Nodes: [matmul_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg9_1, (4096, 11008), (1, 4096), 0), out=buf16)
        del arg9_1
        buf17 = empty_strided_cuda((2048, 11008), (11008, 1), torch.float16)
        # Source Nodes: [matmul_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 4096), (4096, 1), 0), reinterpret_tensor(arg10_1, (4096, 11008), (1, 4096), 0), out=buf17)
        del arg10_1
        buf18 = reinterpret_tensor(buf16, (1, 2048, 11008), (22544384, 11008, 1), 0); del buf16  # reuse
        # Source Nodes: [mul_8, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_3.run(buf18, buf17, 22544384, grid=grid(22544384), stream=stream0)
        del buf17
        buf19 = reinterpret_tensor(buf5, (2048, 4096), (4096, 1), 0); del buf5  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (2048, 11008), (11008, 1), 0), reinterpret_tensor(arg11_1, (11008, 4096), (1, 11008), 0), out=buf19)
        del arg11_1
        del buf18
        buf20 = reinterpret_tensor(buf13, (1, 2048, 4096), (8388608, 4096, 1), 0); del buf13  # reuse
        # Source Nodes: [add_5, x], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf20, arg0_1, buf19, 8388608, grid=grid(8388608), stream=stream0)
        del arg0_1
        del buf19
    return (buf20, buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2048, 4096), (8388608, 4096, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((1, 1, 2048, 128), (262144, 262144, 128, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1, 1, 2048, 128), (262144, 262144, 128, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
