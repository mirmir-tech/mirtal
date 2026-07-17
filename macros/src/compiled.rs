use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, FnArg, GenericArgument, ItemFn, PathArguments, ReturnType, Type, TypeArray,
    parse_macro_input,
};

pub fn expand_macro(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
    match expand(attributes, &function) {
        Ok(expanded) => expanded.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn expand(attributes: TokenStream, function: &ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let shapeless = parse_options(attributes)?;
    let inputs = input_count(function)?;
    let outputs = output_count(function)?;
    let name = &function.sig.ident;
    let compile_name = format_ident!("compile_{name}");
    let visibility = &function.vis;
    let options = if shapeless {
        quote!(::mirtal::CompileOptions::shapeless())
    } else {
        quote!(::mirtal::CompileOptions::default())
    };
    Ok(quote! {
        #function

        #visibility fn #compile_name(
            stream: &::mirtal::Stream,
        ) -> ::mirtal::Result<::mirtal::Compiled<#inputs, #outputs>> {
            stream.compile::<#inputs, #outputs, _>(#options, #name)
        }
    })
}

fn parse_options(attributes: TokenStream) -> syn::Result<bool> {
    if attributes.is_empty() {
        return Ok(false);
    }
    let option = syn::parse::<syn::Ident>(attributes)?;
    if option == "shapeless" {
        Ok(true)
    } else {
        Err(syn::Error::new(option.span(), "expected `shapeless` or no option"))
    }
}

fn input_count(function: &ItemFn) -> syn::Result<&Expr> {
    if function.sig.inputs.len() != 2 {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "compiled graph requires Graph and [Array; N] arguments",
        ));
    }
    let argument = function
        .sig
        .inputs
        .iter()
        .nth(1)
        .ok_or_else(|| syn::Error::new_spanned(&function.sig, "missing array inputs"))?;
    let FnArg::Typed(argument) = argument else {
        return Err(syn::Error::new_spanned(argument, "self is not supported"));
    };
    array_length(&argument.ty)
}

fn output_count(function: &ItemFn) -> syn::Result<&Expr> {
    let ReturnType::Type(_, output) = &function.sig.output else {
        return Err(syn::Error::new_spanned(&function.sig.output, "missing Result output"));
    };
    let Type::Path(result) = output.as_ref() else {
        return Err(syn::Error::new_spanned(output, "expected Result<[Array; N]>"));
    };
    let segment = result
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(result, "empty Result path"))?;
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(syn::Error::new_spanned(segment, "expected Result<[Array; N]>"));
    };
    let output = arguments
        .args
        .first()
        .ok_or_else(|| syn::Error::new_spanned(arguments, "Result has no output"))?;
    let GenericArgument::Type(output) = output else {
        return Err(syn::Error::new_spanned(output, "expected array output"));
    };
    array_length(output)
}

fn array_length(value: &Type) -> syn::Result<&Expr> {
    let Type::Array(TypeArray { len, .. }) = value else {
        return Err(syn::Error::new_spanned(value, "expected [Array; N]"));
    };
    Ok(len)
}
