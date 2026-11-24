//! The Metal source a macro checks at build time and embeds in the binary,
//! shared by `metal_kernel!` and `metal_library!`: both texts come from one
//! join rule, so the source checked is the source shipped.

use std::{env, fs, path::PathBuf};

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    LitStr, Token, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

#[cfg(test)]
mod tests;

/// Ends every file of a `files [...]` list, the last one included.
const FILE_SEPARATOR: &str = "\n";
/// Between the files of a joined source's origin.
const ORIGIN_SEPARATOR: &str = " + ";

mod keyword {
    syn::custom_keyword!(inline);
    syn::custom_keyword!(file);
    syn::custom_keyword!(files);
}

/// Metal source given inline or as files relative to the crate root.
pub enum Source {
    Inline(LitStr),
    /// Files joined in order, each followed by a newline; `file "x"` is the
    /// one-element list.
    Files(Vec<LitStr>),
}

impl Parse for Source {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(keyword::inline) {
            input.parse::<keyword::inline>()?;
            Ok(Self::Inline(input.parse()?))
        } else if lookahead.peek(keyword::files) {
            let mode = input.parse::<keyword::files>()?;
            let content;
            bracketed!(content in input);
            let files = Punctuated::<LitStr, Token![,]>::parse_terminated(&content)?;
            if files.is_empty() {
                return Err(syn::Error::new(mode.span, "`files` requires a path"));
            }
            Ok(Self::Files(files.into_iter().collect()))
        } else if lookahead.peek(keyword::file) {
            input.parse::<keyword::file>()?;
            Ok(Self::Files(vec![input.parse()?]))
        } else {
            Err(lookahead.error())
        }
    }
}

/// A source's text, checked at build time; the expression embedding the same
/// text; and where it came from.
pub struct ResolvedSource {
    pub code: String,
    pub expression: TokenStream,
    pub origin: String,
}

impl Source {
    /// Reads the files; inline source is named `inline_origin`.
    pub fn resolve(&self, inline_origin: &str) -> syn::Result<ResolvedSource> {
        match self {
            Self::Inline(value) => Ok(ResolvedSource {
                code: value.value(),
                expression: quote!(#value),
                origin: inline_origin.into(),
            }),
            Self::Files(values) => Ok(ResolvedSource::join(
                &values.iter().map(ResolvedSource::file).collect::<syn::Result<Vec<_>>>()?,
            )),
        }
    }
}

impl ResolvedSource {
    /// One file relative to the crate root.
    fn file(value: &LitStr) -> syn::Result<Self> {
        let root = env::var_os("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .ok_or_else(|| syn::Error::new(value.span(), "CARGO_MANIFEST_DIR is not set"))?;
        Self::read(&root.join(value.value()), value)
    }

    fn read(path: &std::path::Path, value: &LitStr) -> syn::Result<Self> {
        let code =
            fs::read_to_string(path).map_err(|error| syn::Error::new(value.span(), error))?;
        let absolute = path.to_string_lossy().into_owned();
        Ok(Self {
            code,
            expression: quote!(include_str!(#absolute)),
            origin: absolute,
        })
    }

    /// Files in order, each followed by [`FILE_SEPARATOR`], in both the
    /// checked text and the embedded expression.
    fn join(files: &[Self]) -> Self {
        let code = files.iter().map(|file| file.code.as_str()).collect::<Vec<_>>();
        let parts = files.iter().map(|file| &file.expression);
        let separators = std::iter::repeat(FILE_SEPARATOR);
        let origin = files.iter().map(|file| file.origin.as_str()).collect::<Vec<_>>();
        Self {
            code: code.join(FILE_SEPARATOR) + FILE_SEPARATOR,
            expression: quote!(concat!(#(#parts, #separators),*)),
            origin: origin.join(ORIGIN_SEPARATOR),
        }
    }
}
