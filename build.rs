use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let include = PathBuf::from(
        env::var_os("DEP_MIRTAL_INCLUDE").ok_or("mirtal-sys did not export its include path")?,
    );
    println!("cargo:include={}", include.display());
    Ok(())
}
