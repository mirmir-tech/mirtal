mod syntax;
pub mod validate;

use std::{env, fs, path::PathBuf};

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, LitStr};

use self::{
    syntax::{Buffer, Kernel, Source, TemplateDefault},
    validate::validate,
};

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let descriptor = syn::parse_macro_input!(input as Kernel);
    match descriptor.expand() {
        Ok(output) => output.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

impl Kernel {
    fn expand(self) -> syn::Result<TokenStream> {
        self.validate_contract()?;
        let source = self.source.resolve()?;
        let header = self.header.resolve()?;
        validate(&self, &source.code, &header.code)?;
        let visibility = &self.visibility;
        let function = &self.function;
        let name = &self.name;
        let input_names = self.inputs.iter().map(|buffer| buffer.name.to_string());
        let output_names = self.outputs.iter().map(|buffer| buffer.name.to_string());
        let input_dtypes =
            self.inputs.iter().map(|buffer| dtype_tokens(&self, buffer)).collect::<Vec<_>>();
        let output_dtypes = self
            .outputs
            .iter()
            .map(|buffer| dtype_tokens(&self, buffer))
            .collect::<Vec<_>>();
        let templates = self.templates.iter().map(template_tokens).collect::<Vec<_>>();
        let input_count = self.inputs.len();
        let output_count = self.outputs.len();
        let source_expression = source.expression;
        let source_origin = source.origin;
        let header_expression = header.expression;
        let header_origin = header.origin;
        let row_contiguous = &self.row_contiguous;
        let atomic_outputs = &self.atomic_outputs;
        Ok(quote! {
            #visibility fn #function() -> ::mirtal::Result<
                ::mirtal::MetalKernel<#input_count, #output_count>
            > {
                ::mirtal::MetalKernel::new(::mirtal::KernelDescriptor {
                    name: #name,
                    input_names: [#(#input_names),*],
                    input_dtypes: [#(#input_dtypes),*],
                    output_names: [#(#output_names),*],
                    output_dtypes: [#(#output_dtypes),*],
                    templates: &[#(#templates),*],
                    source: ::mirtal::MetalSource::new(#source_expression, #source_origin),
                    header: ::mirtal::MetalSource::new(#header_expression, #header_origin),
                    row_contiguous: #row_contiguous,
                    atomic_outputs: #atomic_outputs,
                })
            }
        })
    }
}

struct ResolvedSource {
    code: String,
    expression: TokenStream,
    origin: String,
}

impl Source {
    fn resolve(&self) -> syn::Result<ResolvedSource> {
        match self {
            Self::Inline(value) => Ok(ResolvedSource {
                code: value.value(),
                expression: quote!(#value),
                origin: "<inline metal macro>".into(),
            }),
            Self::File(value) => resolve_file(value),
        }
    }
}

fn resolve_file(value: &LitStr) -> syn::Result<ResolvedSource> {
    let root = env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| syn::Error::new(value.span(), "CARGO_MANIFEST_DIR is not set"))?;
    let path = root.join(value.value());
    let code = match fs::read_to_string(&path) {
        Ok(code) => code,
        Err(error) => return Err(syn::Error::new(value.span(), error)),
    };
    let absolute = path.to_string_lossy().into_owned();
    Ok(ResolvedSource {
        code,
        expression: quote!(include_str!(#absolute)),
        origin: absolute,
    })
}

fn dtype_tokens(kernel: &Kernel, buffer: &Buffer) -> TokenStream {
    let element = buffer.element();
    if kernel.dtype_template(element).is_some() {
        let name = element.to_string();
        return quote!(::mirtal::DTypeConstraint::Template(#name));
    }
    let dtype = match element.to_string().as_str() {
        "bool" => quote!(::mirtal::DType::Bool),
        "u32" => quote!(::mirtal::DType::Uint32),
        "i32" => quote!(::mirtal::DType::Int32),
        "f16" => quote!(::mirtal::DType::Float16),
        "bf16" => quote!(::mirtal::DType::Bfloat16),
        "f32" => quote!(::mirtal::DType::Float32),
        "float" => return quote!(::mirtal::DTypeConstraint::Float),
        _ => unreachable!("contract validation rejects unsupported buffer types"),
    };
    quote!(::mirtal::DTypeConstraint::Exact(#dtype))
}

fn template_tokens(template: &syntax::Template) -> TokenStream {
    let name = template.name.to_string();
    let kind = match template.default {
        TemplateDefault::DType(_) => quote!(::mirtal::TemplateKind::DType),
        TemplateDefault::Int(_) => quote!(::mirtal::TemplateKind::Int),
        TemplateDefault::Bool(_) => quote!(::mirtal::TemplateKind::Bool),
    };
    quote!(::mirtal::TemplateParameter { name: #name, kind: #kind })
}

pub fn metal_type(element: &Ident) -> syn::Result<&'static str> {
    match element.to_string().as_str() {
        "bool" => Ok("bool"),
        "u32" => Ok("uint"),
        "i32" => Ok("int"),
        "f16" => Ok("half"),
        "bf16" => Ok("bfloat"),
        "f32" | "float" => Ok("float"),
        _ => Err(syn::Error::new(element.span(), "unsupported Metal type")),
    }
}
