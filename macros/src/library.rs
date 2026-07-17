use std::{env, fs, path::PathBuf};

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Ident, LitStr, Token, Visibility, braced,
    parse::{Parse, ParseStream},
};

use crate::metal::validate::validate_library;

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let library = syn::parse_macro_input!(input as Library);
    match library.expand() {
        Ok(output) => output.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

struct Library {
    visibility: Visibility,
    function: Ident,
    name: LitStr,
    source: Source,
}

enum Source {
    Inline(LitStr),
    File(LitStr),
}

impl Parse for Library {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let function = input.parse()?;
        let content;
        braced!(content in input);
        field(&content, "name")?;
        let name = content.parse()?;
        content.parse::<Token![,]>()?;
        field(&content, "source")?;
        let source = content.parse()?;
        let _ = content.parse::<Option<Token![,]>>()?;
        Ok(Self { visibility, function, name, source })
    }
}

impl Parse for Source {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mode: Ident = input.parse()?;
        let value = input.parse()?;
        match mode.to_string().as_str() {
            "inline" => Ok(Self::Inline(value)),
            "file" => Ok(Self::File(value)),
            _ => Err(syn::Error::new(mode.span(), "expected `inline` or `file`")),
        }
    }
}

impl Library {
    fn expand(self) -> syn::Result<TokenStream> {
        let source = self.source.resolve()?;
        validate_library(&source.code, self.name.span())?;
        let visibility = self.visibility;
        let function = self.function;
        let name = self.name;
        let expression = source.expression;
        let origin = source.origin;
        Ok(quote! {
            #visibility fn #function() -> ::mirtal::Result<::mirtal::MetalLibrary> {
                ::mirtal::MetalLibrary::new(::mirtal::MetalLibraryDescriptor {
                    name: #name,
                    source: ::mirtal::MetalSource::new(#expression, #origin),
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
                origin: "<inline Metal library>".into(),
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

fn field(input: ParseStream<'_>, expected: &str) -> syn::Result<()> {
    let actual: Ident = input.parse()?;
    if actual != expected {
        return Err(syn::Error::new(actual.span(), format!("expected `{expected}`")));
    }
    input.parse::<Token![:]>()?;
    Ok(())
}
