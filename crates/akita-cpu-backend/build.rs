use std::env;
use std::path::PathBuf;
use std::process::Command;

const SOURCE: &str = "src/sources/dense/packing_sme.c";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(akita_packing_sme)");
    println!("cargo:rustc-check-cfg=cfg(akita_fold_metal)");
    println!("cargo:rerun-if-changed={SOURCE}");
    if env::var("TARGET").as_deref() != Ok("aarch64-apple-darwin") {
        return;
    }

    let Some(directory) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return;
    };
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    let Some(deployment) = env::var_os("RUSTC")
        .and_then(|rustc| {
            Command::new(rustc)
                .args([
                    "--target",
                    "aarch64-apple-darwin",
                    "--print",
                    "deployment-target",
                ])
                .output()
                .ok()
        })
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| {
            output
                .trim()
                .strip_prefix("MACOSX_DEPLOYMENT_TARGET=")
                .map(str::to_owned)
        })
    else {
        return;
    };
    let clang_target = format!("--target=arm64-apple-macosx{deployment}");
    let metal_archive = directory.join("libpgl_akita_fold_metal.a");
    let sources = [
        "src/sources/dense/fixed_fold.m",
        "src/sources/quartic.m",
        "src/sources/quad_lookup.m",
    ];
    let mut archive = Command::new("xcrun");
    archive
        .args(["--sdk", "macosx", "ar", "crs"])
        .arg(&metal_archive);
    let mut compiled = true;
    for source in sources {
        println!("cargo:rerun-if-changed={source}");
        let object = directory.join(
            std::path::Path::new(source)
                .with_extension("o")
                .file_name()
                .unwrap(),
        );
        compiled &= Command::new("xcrun")
            .args([
                "--sdk",
                "macosx",
                "clang",
                &clang_target,
                "-fobjc-arc",
                "-O3",
                "-c",
                source,
                "-o",
            ])
            .arg(&object)
            .status()
            .is_ok_and(|s| s.success());
        archive.arg(object);
    }
    if compiled && archive.status().is_ok_and(|s| s.success()) {
        println!("cargo:rustc-cfg=akita_fold_metal");
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=static=pgl_akita_fold_metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
    }
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
    let library = directory.join("libpgl_akita_packing_sme.a");
    let Ok(compile) = Command::new("xcrun")
        .args([
            "--sdk",
            "macosx",
            "clang",
            &clang_target,
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
        println!("cargo:rustc-link-lib=static=pgl_akita_packing_sme");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
