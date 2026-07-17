use syn::{
    Ident, LitBool, LitInt, LitStr, Token, Visibility, braced, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

pub(super) struct Kernel {
    pub visibility: Visibility,
    pub function: Ident,
    pub name: LitStr,
    pub templates: Vec<Template>,
    pub inputs: Vec<Buffer>,
    pub outputs: Vec<Buffer>,
    pub source: Source,
    pub header: Source,
    pub row_contiguous: LitBool,
    pub atomic_outputs: LitBool,
}

pub(super) struct Template {
    pub name: Ident,
    pub default: TemplateDefault,
}

pub(super) enum TemplateDefault {
    DType(Ident),
    Int(LitInt),
    Bool(LitBool),
}

pub(super) struct Buffer {
    pub name: Ident,
    pub kind: BufferKind,
}

pub(super) enum BufferKind {
    Array(Ident),
    Scalar(Ident),
}

pub(super) enum Source {
    Inline(LitStr),
    File(LitStr),
}

impl Parse for Kernel {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let function = input.parse()?;
        let content;
        braced!(content in input);
        field(&content, "name")?;
        let name = content.parse()?;
        comma(&content)?;
        field(&content, "templates")?;
        let templates = list(&content)?;
        comma(&content)?;
        field(&content, "inputs")?;
        let inputs = list(&content)?;
        comma(&content)?;
        field(&content, "outputs")?;
        let outputs = list(&content)?;
        comma(&content)?;
        field(&content, "source")?;
        let source = content.parse()?;
        comma(&content)?;
        field(&content, "header")?;
        let header = content.parse()?;
        comma(&content)?;
        field(&content, "row_contiguous")?;
        let row_contiguous = content.parse()?;
        comma(&content)?;
        field(&content, "atomic_outputs")?;
        let atomic_outputs = content.parse()?;
        let _ = content.parse::<Option<Token![,]>>()?;
        Ok(Self {
            visibility,
            function,
            name,
            templates,
            inputs,
            outputs,
            source,
            header,
            row_contiguous,
            atomic_outputs,
        })
    }
}

impl Parse for Template {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let kind: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let default = match kind.to_string().as_str() {
            "dtype" => TemplateDefault::DType(input.parse()?),
            "int" => TemplateDefault::Int(input.parse()?),
            "bool" => TemplateDefault::Bool(input.parse()?),
            _ => return Err(syn::Error::new(kind.span(), "expected `dtype`, `int`, or `bool`")),
        };
        Ok(Self { name, default })
    }
}

impl Parse for Buffer {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let marker: Ident = input.parse()?;
        let kind = if marker == "scalar" {
            input.parse::<Token![<]>()?;
            let element = input.parse()?;
            input.parse::<Token![>]>()?;
            BufferKind::Scalar(element)
        } else {
            BufferKind::Array(marker)
        };
        Ok(Self { name, kind })
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

impl Kernel {
    pub(super) fn validate_contract(&self) -> syn::Result<()> {
        if self.inputs.is_empty() || self.outputs.is_empty() {
            return Err(syn::Error::new(self.name.span(), "kernel requires inputs and outputs"));
        }
        for buffer in self.inputs.iter().chain(&self.outputs) {
            let element = buffer.element();
            if primitive(element) || self.dtype_template(element).is_some() {
                continue;
            }
            return Err(syn::Error::new(
                element.span(),
                "buffer type must be a primitive dtype or declared dtype template",
            ));
        }
        Ok(())
    }

    pub(super) fn dtype_template(&self, name: &Ident) -> Option<&Template> {
        self.templates.iter().find(|template| {
            template.name == *name && matches!(template.default, TemplateDefault::DType(_))
        })
    }
}

impl Buffer {
    pub(super) const fn element(&self) -> &Ident {
        match &self.kind {
            BufferKind::Array(element) | BufferKind::Scalar(element) => element,
        }
    }

    pub(super) const fn is_scalar(&self) -> bool {
        matches!(self.kind, BufferKind::Scalar(_))
    }
}

fn list<T: Parse>(input: ParseStream<'_>) -> syn::Result<Vec<T>> {
    let content;
    bracketed!(content in input);
    Ok(Punctuated::<T, Token![,]>::parse_terminated(&content)?.into_iter().collect())
}

fn field(input: ParseStream<'_>, expected: &str) -> syn::Result<()> {
    let actual: Ident = input.parse()?;
    if actual != expected {
        return Err(syn::Error::new(actual.span(), format!("expected `{expected}`")));
    }
    input.parse::<Token![:]>()?;
    Ok(())
}

fn comma(input: ParseStream<'_>) -> syn::Result<()> {
    input.parse::<Token![,]>()?;
    Ok(())
}

fn primitive(element: &Ident) -> bool {
    matches!(
        element.to_string().as_str(),
        "bool" | "u32" | "i32" | "f16" | "bf16" | "f32" | "float"
    )
}
