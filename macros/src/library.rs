use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Ident, LitStr, Token, Visibility, braced,
    parse::{Parse, ParseStream},
};

use crate::{metal::validate::validate_library, source::Source};

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

impl Library {
    fn expand(self) -> syn::Result<TokenStream> {
        let source = self.source.resolve("<inline Metal library>")?;
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

fn field(input: ParseStream<'_>, expected: &str) -> syn::Result<()> {
    let actual: Ident = input.parse()?;
    if actual != expected {
        return Err(syn::Error::new(actual.span(), format!("expected `{expected}`")));
    }
    input.parse::<Token![:]>()?;
    Ok(())
}
