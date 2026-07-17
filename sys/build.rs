use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
};

fn main() -> Result<(), Box<dyn Error>> {
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?);
    println!("cargo:include={}", manifest.join("native/include").display());
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/attention.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/ops.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/aliasing.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/prepared_metal.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/quantized.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/read.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/bridge/rope.h");
    println!("cargo:rerun-if-changed=native/include/mirtal/native.h");
    println!("cargo:rerun-if-changed=native/src/bridge.cpp");
    println!("cargo:rerun-if-changed=native/src/compile.cpp");
    println!("cargo:rerun-if-changed=native/src/metal.cpp");
    println!("cargo:rerun-if-changed=native/src/prepared_metal.cpp");
    println!("cargo:rerun-if-changed=native/src/graph.cpp");
    println!("cargo:rerun-if-changed=native/src/io.cpp");
    println!("cargo:rerun-if-changed=native/src/ops.cpp");
    println!("cargo:rerun-if-changed=native/src/aliasing.cpp");
    println!("cargo:rerun-if-changed=native/src/quantized.cpp");
    println!("cargo:rerun-if-changed=native/src/read.cpp");
    println!("cargo:rerun-if-changed=native/src/rope.cpp");
    let prefix = mlx_prefix()?;
    let include = prefix.join("include");
    let library = prefix.join("lib");
    require(&include.join("mlx/mlx.h"))?;
    require(&library.join("libmlx.dylib"))?;
    require(&library.join("mlx.metallib"))?;
    cxx_build::bridges([
        "src/lib.rs",
        "src/graph.rs",
        "src/io.rs",
        "src/ops.rs",
        "src/aliasing.rs",
        "src/read.rs",
    ])
    .file("native/src/bridge.cpp")
    .file("native/src/attention.cpp")
    .file("native/src/compile.cpp")
    .file("native/src/graph.cpp")
    .file("native/src/io.cpp")
    .file("native/src/metal.cpp")
    .file("native/src/prepared_metal.cpp")
    .file("native/src/ops.cpp")
    .file("native/src/aliasing.cpp")
    .file("native/src/quantized.cpp")
    .file("native/src/read.cpp")
    .file("native/src/rope.cpp")
    .include("native/include")
    .include(&include)
    .include(include.join("metal_cpp"))
    .std("c++20")
    .flag_if_supported("-Wno-deprecated-copy")
    .flag_if_supported("-Wno-unused-parameter")
    .warnings(true)
    .compile("mirtal_bridge");
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library.display());
    Ok(())
}

fn mlx_prefix() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(prefix) = env::var_os("MLX_PREFIX") {
        return Ok(PathBuf::from(prefix));
    }
    ["/opt/homebrew/opt/mlx", "/usr/local/opt/mlx"]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .ok_or_else(|| "MLX was not found; set MLX_PREFIX".into())
}

fn require(path: &Path) -> Result<(), Box<dyn Error>> {
    path.exists()
        .then_some(())
        .ok_or_else(|| format!("required MLX path does not exist: {}", path.display()).into())
}
