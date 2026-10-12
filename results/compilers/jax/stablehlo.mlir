module @jit_graph attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024xf16> {mhlo.sharding = "{replicated}"}) -> (tensor<1024x1024xf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1024x1024xf16>) -> tensor<1024x1024xf32>
    %cst = stablehlo.constant dense<0.699999988> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x1024xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x1024xf32>
    %cst_0 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x1024xf32>
    %4 = stablehlo.add %2, %3 : tensor<1024x1024xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_1) applies stablehlo.maximum across dimensions = [1] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<1024xf32>) -> tensor<1024x1xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1024x1xf32>) -> tensor<1024x1024xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1024xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1024xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<1024xf32>) -> tensor<1024x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1024x1xf32>) -> tensor<1024x1024xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1024xf32>
    %14 = stablehlo.convert %13 : (tensor<1024x1024xf32>) -> tensor<1024x1024xf16>
    return %14 : tensor<1024x1024xf16>
  }
}
