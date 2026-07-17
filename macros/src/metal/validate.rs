use std::{
    fmt::Write as _,
    io::Write,
    process::{Command, Stdio},
};

use proc_macro2::Span;

use super::{
    metal_type,
    syntax::{Buffer, Kernel, TemplateDefault},
};

pub(super) fn validate(kernel: &Kernel, source: &str, header: &str) -> syn::Result<()> {
    let wrapper = wrapper(kernel, source, header)?;
    compile(&wrapper, kernel.name.span())
}

pub fn validate_library(source: &str, span: Span) -> syn::Result<()> {
    compile(source, span)
}

fn compile(source: &str, span: Span) -> syn::Result<()> {
    let mut child = match Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-x", "metal", "-fsyntax-only", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return Err(syn::Error::new(span, error)),
    };
    let Some(mut input) = child.stdin.take() else {
        return Err(syn::Error::new(span, "Metal compiler stdin is unavailable"));
    };
    if let Err(error) = input.write_all(source.as_bytes()) {
        return Err(syn::Error::new(span, error));
    }
    drop(input);
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(error) => return Err(syn::Error::new(span, error)),
    };
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    Err(syn::Error::new(span, format!("Metal syntax validation failed:\n{diagnostics}")))
}

fn wrapper(kernel: &Kernel, source: &str, header: &str) -> syn::Result<String> {
    let mut arguments = Vec::with_capacity(kernel.inputs.len() + kernel.outputs.len() + 6);
    for (index, buffer) in kernel.inputs.iter().enumerate() {
        let element = buffer_type(kernel, buffer)?;
        let argument = if buffer.is_scalar() {
            format!("constant {element}& {} [[buffer({index})]]", buffer.name)
        } else {
            format!("const device {element}* {} [[buffer({index})]]", buffer.name)
        };
        arguments.push(argument);
    }
    for (offset, buffer) in kernel.outputs.iter().enumerate() {
        let index = kernel.inputs.len() + offset;
        arguments.push(format!(
            "device {}* {} [[buffer({index})]]",
            buffer_type(kernel, buffer)?,
            buffer.name
        ));
    }
    arguments.extend([
        "uint3 thread_position_in_grid [[thread_position_in_grid]]".into(),
        "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]".into(),
        "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]".into(),
        "uint3 threads_per_threadgroup [[threads_per_threadgroup]]".into(),
        "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]".into(),
        "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]".into(),
        "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]".into(),
    ]);
    let templates = validation_templates(kernel)?;
    Ok(format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{templates}\n{header}\nkernel void __mirtal_validate(\n{}\n) {{\n{source}\n}}\n",
        arguments.join(",\n")
    ))
}

fn buffer_type(kernel: &Kernel, buffer: &Buffer) -> syn::Result<&'static str> {
    if let Some(template) = kernel.dtype_template(buffer.element()) {
        let TemplateDefault::DType(dtype) = &template.default else {
            unreachable!("dtype_template only returns dtype templates");
        };
        return metal_type(dtype);
    }
    metal_type(buffer.element())
}

fn validation_templates(kernel: &Kernel) -> syn::Result<String> {
    let mut output = String::new();
    for template in &kernel.templates {
        let value = match &template.default {
            TemplateDefault::DType(dtype) => metal_type(dtype)?.into(),
            TemplateDefault::Int(value) => value.base10_digits().into(),
            TemplateDefault::Bool(value) => value.value.to_string(),
        };
        if writeln!(&mut output, "#define {} {value}", template.name).is_err() {
            return Err(syn::Error::new(template.name.span(), "template preamble is invalid"));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_inline_metal_during_rust_compilation() -> syn::Result<()> {
        let kernel = syn::parse_str::<Kernel>(
            r#"fn broken {
                name: "broken",
                templates: [],
                inputs: [input: f32],
                outputs: [output: f32],
                source: inline "output[0] = ;",
                header: inline "",
                row_contiguous: true,
                atomic_outputs: false,
            }"#,
        )?;
        let result = validate(&kernel, "output[0] = ;", "");
        let Err(error) = result else {
            return Err(syn::Error::new(kernel.name.span(), "invalid Metal was accepted"));
        };
        assert!(error.to_string().contains("Metal syntax validation failed"));
        Ok(())
    }
}
