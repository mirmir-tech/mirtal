use quote::quote;
use syn::LitStr;

use super::{ResolvedSource, Source};

/// The checked text is exactly what the embedded `concat!` expands to: each
/// file's text, each followed by the separator.
#[test]
fn joins_the_checked_and_the_embedded_text_alike() -> syn::Result<()> {
    let directory = std::env::temp_dir().join(format!("mirtal-macros-join-{}", std::process::id()));
    std::fs::create_dir_all(&directory)
        .map_err(|error| syn::Error::new(proc_macro2::Span::call_site(), error))?;
    let paths = [directory.join("a.h"), directory.join("b.metal")];
    let texts = ["a", "b\n"];
    for (path, text) in paths.iter().zip(texts) {
        std::fs::write(path, text)
            .map_err(|error| syn::Error::new(proc_macro2::Span::call_site(), error))?;
    }
    let literal = LitStr::new("unused", proc_macro2::Span::call_site());
    let files = paths
        .iter()
        .map(|path| ResolvedSource::read(path, &literal))
        .collect::<syn::Result<Vec<_>>>();
    let _removed = std::fs::remove_dir_all(&directory);
    let joined = ResolvedSource::join(&files?);

    assert_eq!(joined.code, "a\nb\n\n");
    let [first, second] = paths.map(|path| path.to_string_lossy().into_owned());
    let expected = quote!(concat!(include_str!(#first), "\n", include_str!(#second), "\n"));
    assert_eq!(joined.expression.to_string(), expected.to_string());
    assert_eq!(joined.origin, format!("{first} + {second}"));
    Ok(())
}

#[test]
fn parses_the_source_forms() -> syn::Result<()> {
    assert!(matches!(syn::parse_str::<Source>(r#"inline "kernel""#)?, Source::Inline(_)));
    assert!(
        matches!(syn::parse_str::<Source>(r#"file "a.metal""#)?, Source::Files(files) if files.len() == 1)
    );
    assert!(
        matches!(syn::parse_str::<Source>(r#"files ["a.h", "b.metal"]"#)?, Source::Files(files) if files.len() == 2)
    );
    assert!(syn::parse_str::<Source>("files []").is_err());
    assert!(syn::parse_str::<Source>(r#"path "a.metal""#).is_err());
    Ok(())
}
