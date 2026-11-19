use std::env;
use std::path::PathBuf;
use std::process::Command;

const SOURCE: &str = "src/sources/dense/packing_sme.c";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(akita_packing_sme)");
    println!("cargo:rerun-if-changed={SOURCE}");
    if env::var("TARGET").as_deref() != Ok("aarch64-apple-darwin") {
        return;
    }

    let Some(directory) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return;
    };
    let Ok(resource_dir) = Command::new("xcrun")
        .args(["--sdk", "macosx", "clang", "-print-resource-dir"])
        .output()
    else {
        return;
    };
    if !resource_dir.status.success() {
        return;
    }
    let resource_dir = String::from_utf8_lossy(&resource_dir.stdout);
    let runtime = PathBuf::from(resource_dir.trim()).join("lib/darwin");
    if !runtime.join("libclang_rt.osx.a").exists() {
        return;
    }

    let object = directory.join("packing_sme.o");
    let library = directory.join("libakita_packing_sme.a");
    let Ok(compile) = Command::new("xcrun")
        .args([
            "--sdk",
            "macosx",
            "clang",
            "--target=arm64-apple-darwin",
            "-O3",
            "-march=armv8-a+sme+sme-i16i64",
            "-c",
            SOURCE,
            "-o",
        ])
        .arg(&object)
        .output()
    else {
        return;
    };
    if !compile.status.success() {
        return;
    }
    let Ok(archive) = Command::new("xcrun")
        .args(["--sdk", "macosx", "ar", "crs"])
        .arg(&library)
        .arg(&object)
        .status()
    else {
        return;
    };
    if archive.success() {
        println!("cargo:rustc-cfg=akita_packing_sme");
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-search=native={}", runtime.display());
        println!("cargo:rustc-link-lib=static=akita_packing_sme");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
