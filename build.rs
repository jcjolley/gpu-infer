use std::path::PathBuf;

fn main() {
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| {
            // Common locations
            for path in &["/opt/cuda", "/usr/local/cuda"] {
                if std::path::Path::new(path).exists() {
                    return path.to_string();
                }
            }
            panic!("CUDA not found. Set CUDA_HOME env var.");
        });

    let flashinfer_include: PathBuf =
        ["third_party", "flashinfer", "include"].iter().collect();

    let cutlass_include: PathBuf = [
        "third_party", "flashinfer", "3rdparty", "cutlass", "include",
    ]
    .iter()
    .collect();

    cc::Build::new()
        .cuda(true)
        .file("kernels/wrapper.cu")
        .include(&flashinfer_include)
        .include(&cutlass_include)
        .include(format!("{}/include", cuda_path))
        .include("kernels")
        // Target RTX 4090 (compute 8.9)
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-std=c++17")
        .flag("-O2")
        // Suppress excessive warnings from template-heavy code
        .flag("-w")
        // FlashInfer uses extended lambda
        .flag("--expt-extended-lambda")
        .flag("--expt-relaxed-constexpr")
        .compile("flashinfer_wrapper");

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=kernels/wrapper.cu");
    println!("cargo:rerun-if-changed=kernels/wrapper.h");
}
