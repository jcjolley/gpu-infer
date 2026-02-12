//! gpu-infer spike: Prove we can call FlashInfer attention kernels from Rust via FFI.

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use gpu_infer::attention::ffi::flashinfer_single_decode_f16;
use half::f16;

fn main() {
    let num_qo_heads: i32 = 32;
    let num_kv_heads: i32 = 8; // GQA 4:1
    let head_dim: i32 = 128;
    let seq_len: i32 = 64;

    println!("gpu-infer spike: FlashInfer single-decode attention");
    println!("  Q heads:    {}", num_qo_heads);
    println!("  KV heads:   {} (GQA {}:1)", num_kv_heads, num_qo_heads / num_kv_heads);
    println!("  Head dim:   {}", head_dim);
    println!("  Seq len:    {}", seq_len);
    println!();

    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
    println!("GPU: {}", ctx.name().unwrap_or("unknown".into()));

    let stream = ctx.default_stream();

    let q_len = (num_qo_heads * head_dim) as usize;
    let kv_len = (seq_len * num_kv_heads * head_dim) as usize;
    let o_len = (num_qo_heads * head_dim) as usize;

    // Build fp16 data as u16 (same repr)
    let q_host: Vec<u16> = (0..q_len)
        .map(|i| f16::from_f32(0.01 * ((i % 128) as f32 - 64.0)).to_bits())
        .collect();
    let k_host: Vec<u16> = (0..kv_len)
        .map(|i| f16::from_f32(0.01 * ((i % 128) as f32 - 64.0)).to_bits())
        .collect();
    let v_host: Vec<u16> = (0..kv_len)
        .map(|i| f16::from_f32(0.01 * ((i % 128) as f32)).to_bits())
        .collect();

    // Copy to GPU
    let q_dev = stream.clone_htod(&q_host).expect("Failed to copy Q to GPU");
    let k_dev = stream.clone_htod(&k_host).expect("Failed to copy K to GPU");
    let v_dev = stream.clone_htod(&v_host).expect("Failed to copy V to GPU");
    let mut o_dev: CudaSlice<u16> = stream.alloc_zeros(o_len).expect("Failed to alloc output");

    println!("Tensors allocated on GPU ({:.1} KB total)",
        (q_len + kv_len * 2 + o_len) as f64 * 2.0 / 1024.0);
    println!();

    // Get raw device pointers
    let (q_ptr, _q_guard) = q_dev.device_ptr(&stream);
    let (k_ptr, _k_guard) = k_dev.device_ptr(&stream);
    let (v_ptr, _v_guard) = v_dev.device_ptr(&stream);
    let (o_ptr, _o_guard) = o_dev.device_ptr_mut(&stream);

    print!("Calling FlashInfer SingleDecodeWithKVCache... ");
    let err = unsafe {
        flashinfer_single_decode_f16(
            q_ptr as *const std::ffi::c_void,
            k_ptr as *const std::ffi::c_void,
            v_ptr as *const std::ffi::c_void,
            o_ptr as *mut std::ffi::c_void,
            num_qo_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            std::ptr::null_mut(), // default stream
        )
    };

    drop(_q_guard);
    drop(_k_guard);
    drop(_v_guard);
    drop(_o_guard);

    if err != 0 {
        eprintln!("FAILED (CUDA error {})", err);
        std::process::exit(1);
    }

    stream.synchronize().expect("CUDA synchronize failed");
    println!("OK");

    // Copy output back
    let o_host = stream.clone_dtoh(&o_dev).expect("Failed to copy output from GPU");

    // Interpret u16 as f16
    let non_zero = o_host.iter().filter(|&&bits| bits != 0).count();
    let values: Vec<f32> = o_host.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect();
    let max_val = values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let has_nan = values.iter().any(|v| v.is_nan());

    println!();
    println!("Results:");
    println!("  Output shape: [{}, {}]", num_qo_heads, head_dim);
    println!("  Non-zero elements: {}/{}", non_zero, o_len);
    println!("  Max absolute value: {:.6}", max_val);
    println!("  Contains NaN: {}", has_nan);
    println!("  First 8 values: {:?}", &values[..8]);
    println!();

    if non_zero > 0 && !has_nan {
        println!("SPIKE PASSED: FlashInfer attention kernel executed successfully from Rust!");
    } else if has_nan {
        eprintln!("SPIKE FAILED: Output contains NaN values");
        std::process::exit(1);
    } else {
        eprintln!("SPIKE FAILED: Output is all zeros");
        std::process::exit(1);
    }
}
