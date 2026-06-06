use std::borrow::Cow;

use super::markdown;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct DocstringSections<'a> {
    parameters: Vec<DocstringItem<'a>>,
    attributes: Vec<DocstringItem<'a>>,
    returns: Vec<DocstringItem<'a>>,
    raises: Vec<DocstringItem<'a>>,
}

impl<'a> DocstringSections<'a> {
    pub(super) fn push(&mut self, kind: DocstringSectionKind, item: DocstringItem<'a>) {
        match kind {
            DocstringSectionKind::Parameters => self.parameters.push(item),
            DocstringSectionKind::Attributes => self.attributes.push(item),
            DocstringSectionKind::Returns => self.returns.push(item),
            DocstringSectionKind::Raises => self.raises.push(item),
        }
    }

    pub(super) fn render_markdown(&self) -> String {
        let mut output = String::new();
        render_markdown_section(&mut output, "Parameters", &self.parameters);
        render_markdown_section(&mut output, "Attributes", &self.attributes);
        render_markdown_section(&mut output, "Returns", &self.returns);
        render_markdown_section(&mut output, "Raises", &self.raises);
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DocstringSectionKind {
    Parameters,
    Attributes,
    Returns,
    Raises,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DocstringItem<'a> {
    name: Option<&'a str>,
    ty: Option<&'a str>,
    description: &'a str,
}

impl<'a> DocstringItem<'a> {
    pub(super) fn new(name: Option<&'a str>, ty: Option<&'a str>, description: &'a str) -> Self {
        Self {
            name,
            ty,
            description,
        }
    }

    fn render_into(&self, output: &mut String) {
        let mut has_label = false;

        if let Some(name) = self.name {
            output.push_str(&markdown_code_span(name));
            has_label = true;
        }

        if let Some(ty) = self.ty
            && !ty.is_empty()
        {
            if has_label {
                output.push(' ');
                output.push('(');
                output.push_str(&markdown_type_code_span(ty));
                output.push(')');
            } else {
                output.push_str(&markdown_type_code_span(ty));
                has_label = true;
            }
        }

        if !self.description.is_empty() {
            let starts_with_block = description_starts_with_markdown_block(self.description);
            if has_label {
                output.push_str(if starts_with_block { ":\n" } else { ": " });
            }
            if starts_with_block {
                output.push_str(self.description);
            } else {
                output.push_str(&self.description.replace('\n', "\n    "));
            }
        }
    }
}

fn render_markdown_section(output: &mut String, heading: &str, fields: &[DocstringItem<'_>]) {
    if fields.is_empty() {
        return;
    }

    if !output.is_empty() {
        output.push_str("\n\n");
    }

    output.push_str("## ");
    output.push_str(heading);
    output.push('\n');

    let mut previous_description = None;
    for field in fields {
        if previous_description.is_some() {
            output.push('\n');
            if previous_description.is_some_and(description_leaves_doctest_open) {
                output.push('\n');
            }
        }

        field.render_into(output);
        previous_description = Some(field.description);
    }
}

fn description_leaves_doctest_open(description: &str) -> bool {
    let mut markdown_fence: Option<markdown::MarkdownFence<'_>> = None;
    let mut in_doctest = false;

    for line in description.lines().map(|line| line.trim_start_matches(' ')) {
        if let Some(fence) = markdown_fence {
            if fence.is_closed_by(line) {
                markdown_fence = None;
            }
        } else if in_doctest {
            if line.is_empty() {
                in_doctest = false;
            }
        } else if line.starts_with(">>>") {
            in_doctest = true;
        } else if let Some(fence) = markdown::MarkdownFence::find(line) {
            markdown_fence = Some(fence);
        }
    }

    in_doctest
}

fn description_starts_with_markdown_block(description: &str) -> bool {
    description.lines().next().is_some_and(|first_line| {
        let first_line = first_line.trim_start_matches(' ');
        markdown::MarkdownFence::find(first_line).is_some() || first_line.starts_with(">>>")
    })
}

fn markdown_type_code_span(ty: &str) -> String {
    let normalized = normalized_type(ty);
    markdown_code_span(&normalized)
}

fn normalized_type(ty: &str) -> Cow<'_, str> {
    if !ty.contains('\n') {
        return Cow::Borrowed(ty);
    }

    let mut normalized = String::new();
    for line in ty.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        normalized.push_str(line);
    }

    Cow::Owned(normalized)
}

fn markdown_code_span(text: &str) -> String {
    let longest_backtick_run = text
        .split(|char| char != '`')
        .map(str::len)
        .max()
        .unwrap_or(0);
    let delimiter = "`".repeat(longest_backtick_run + 1);
    if text.starts_with('`') || text.ends_with('`') {
        format!("{delimiter} {text} {delimiter}")
    } else {
        format!("{delimiter}{text}{delimiter}")
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;

    use super::{DocstringItem, DocstringSectionKind, DocstringSections};

    #[test]
    fn sections_render_in_canonical_order() {
        let mut sections = DocstringSections::default();
        sections.push(
            DocstringSectionKind::Raises,
            DocstringItem::new(Some("ValueError"), None, "Invalid value."),
        );
        sections.push(
            DocstringSectionKind::Parameters,
            DocstringItem::new(Some("value"), Some("str"), "The value."),
        );
        sections.push(
            DocstringSectionKind::Returns,
            DocstringItem::new(None, Some("bool"), "Whether validation passed."),
        );
        sections.push(
            DocstringSectionKind::Attributes,
            DocstringItem::new(Some("cache"), Some("dict[str,\n object]"), "Cached data."),
        );

        assert_snapshot!(sections.render_markdown(), @"
        ## Parameters
        `value` (`str`): The value.

        ## Attributes
        `cache` (`dict[str, object]`): Cached data.

        ## Returns
        `bool`: Whether validation passed.

        ## Raises
        `ValueError`: Invalid value.
        ");
    }

    #[test]
    fn sections_render_multiline_and_block_descriptions() {
        let mut sections = DocstringSections::default();
        sections.push(
            DocstringSectionKind::Parameters,
            DocstringItem::new(
                Some("`value`"),
                None,
                "First sentence.\nContinued sentence.\n\nSecond paragraph.",
            ),
        );
        sections.push(
            DocstringSectionKind::Parameters,
            DocstringItem::new(
                Some("example"),
                None,
                "```python\nif ok:\n    do_work()\n```",
            ),
        );
        sections.push(
            DocstringSectionKind::Parameters,
            DocstringItem::new(Some("prompt"), None, ">>> print('prompt')"),
        );
        sections.push(
            DocstringSectionKind::Parameters,
            DocstringItem::new(Some("other"), None, "Another parameter."),
        );

        assert_snapshot!(sections.render_markdown(), @r#"
        ## Parameters
        `` `value` ``: First sentence.
            Continued sentence.
            
            Second paragraph.
        `example`:
        ```python
        if ok:
            do_work()
        ```
        `prompt`:
        >>> print('prompt')

        `other`: Another parameter.
        "#);
    }
}
