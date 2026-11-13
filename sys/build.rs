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
    println!("cargo:rerun-if-changed=native/src/attention.cpp");
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
    println!("cargo:rerun-if-changed=native/src/memory.cpp");
    let prefix = mlx_prefix()?;
    let include = prefix.join("include");
    let library = prefix.join("lib");
    println!("cargo:rerun-if-changed={}", include.join("mlx/version.h").display());
    println!("cargo:rerun-if-changed={}", library.join("libmlx.dylib").display());
    require(&include.join("mlx/mlx.h"))?;
    require(&library.join("libmlx.dylib"))?;
    require(&library.join("mlx.metallib"))?;
    require_version(&include.join("mlx/version.h"))?;
    cxx_build::bridges([
        "src/lib.rs",
        "src/graph.rs",
        "src/io.rs",
        "src/memory.rs",
        "src/ops.rs",
        "src/aliasing.rs",
        "src/read.rs",
        "src/quantized.rs",
    ])
    .file("native/src/bridge.cpp")
    .file("native/src/attention.cpp")
    .file("native/src/compile.cpp")
    .file("native/src/graph.cpp")
    .file("native/src/io.cpp")
    .file("native/src/metal.cpp")
    .file("native/src/prepared_metal.cpp")
    .file("native/src/memory.cpp")
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

/// The oldest MLX whose headers the bridge compiles against: 0.32.4 made
/// `allocator::Buffer` data an accessor (`Data::buffer()`).
const MINIMUM_MLX: (u32, u32, u32) = (0, 32, 4);

fn require_version(header: &Path) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(header)?;
    let component = |name: &str| -> Result<u32, Box<dyn Error>> {
        let prefix = format!("#define MLX_VERSION_{name} ");
        let value = text
            .lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .ok_or_else(|| format!("{} does not define MLX_VERSION_{name}", header.display()))?;
        Ok(value.trim().parse()?)
    };
    let found = (component("MAJOR")?, component("MINOR")?, component("PATCH")?);
    if found < MINIMUM_MLX {
        let (major, minor, patch) = found;
        let (want_major, want_minor, want_patch) = MINIMUM_MLX;
        return Err(format!(
            "MLX {major}.{minor}.{patch} at {} is too old; mirtal needs MLX \
             {want_major}.{want_minor}.{want_patch} or later (set MLX_PREFIX)",
            header.display()
        )
        .into());
    }
    Ok(())
}
