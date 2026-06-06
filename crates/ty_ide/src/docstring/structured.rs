use std::borrow::Cow;
use std::ops::Range;

use ruff_python_trivia::leading_indentation;
use ruff_source_file::UniversalNewlines;

use super::rest::PreformattedBlockScanner;
use super::sections::{DocstringItem, DocstringSectionKind, DocstringSections};

pub(super) struct Docstring<'a> {
    raw: &'a str,
    sections: Vec<Section>,
}

impl<'a> Docstring<'a> {
    pub(super) fn parse(raw: &'a str, style: Style) -> Self {
        Self {
            raw,
            sections: parse_sections(raw, style),
        }
    }

    pub(super) fn render_markdown(&self) -> Cow<'a, str> {
        let mut output: Option<String> = None;
        let mut rendered_through = 0;

        for section in &self.sections {
            if section.indent != 0 || section.range.start < rendered_through {
                continue;
            }

            let markdown = section.render_markdown();
            if markdown.is_empty() {
                continue;
            }

            let output = output.get_or_insert_with(String::new);
            output.push_str(&self.raw[rendered_through..section.range.start]);
            output.push_str(&markdown);
            rendered_through = section.range.end;
        }

        let Some(mut output) = output else {
            return Cow::Borrowed(self.raw);
        };
        output.push_str(&self.raw[rendered_through..]);
        Cow::Owned(output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Style {
    Google,
    Numpy,
}

#[derive(Debug, Clone, Copy)]
struct Line<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Section {
    kind: SectionKind,
    indent: usize,
    range: Range<usize>,
    items: Vec<Item>,
}

impl Section {
    fn render_markdown(&self) -> String {
        let mut sections = DocstringSections::default();
        for item in &self.items {
            sections.push(
                self.kind.docstring_section_kind(),
                DocstringItem::new(
                    item.display_name.as_deref(),
                    item.ty.as_deref(),
                    item.description.as_str(),
                ),
            );
        }
        sections.render_markdown()
    }
}

fn parse_sections(raw: &str, style: Style) -> Vec<Section> {
    let lines = raw
        .universal_newlines()
        .map(|line| Line {
            text: line.as_str(),
            start: line.start().to_usize(),
            end: line.end().to_usize(),
        })
        .collect::<Vec<_>>();

    let mut sections = Vec::new();
    let mut preformatted_blocks = PreformattedBlockScanner::default();
    let mut index = 0;

    while index < lines.len() {
        if preformatted_blocks.consume_preformatted_line(lines[index].text) {
            index += 1;
            continue;
        }

        let Some(header) = parse_section_header(style, &lines, index) else {
            preformatted_blocks.observe_non_preformatted_line(lines[index].text);
            index += 1;
            continue;
        };

        let mut body_end = header.body_start;
        let mut range_end = header.range_end;
        let mut body_preformatted_blocks = PreformattedBlockScanner::default();
        while let Some(line) = lines.get(body_end) {
            if body_preformatted_blocks.consume_preformatted_line(line.text) {
                range_end = line.end;
                body_end += 1;
                continue;
            }

            let previous_body = &lines[header.body_start..body_end];
            if line.text.trim().is_empty()
                && !blank_line_continues_section(style, previous_body, &lines[body_end..], header)
            {
                break;
            }

            if parse_section_header(style, &lines, body_end)
                .is_some_and(|next| next.indent <= header.indent)
            {
                break;
            }

            if !line.text.trim().is_empty()
                && !line_belongs_to_body(style, line, previous_body, &lines[body_end + 1..], header)
            {
                break;
            }

            body_preformatted_blocks.observe_non_preformatted_line(line.text);
            range_end = line.end;
            body_end += 1;
        }

        if let Some(items) = parse_items(style, header.kind, &lines[header.body_start..body_end]) {
            sections.push(Section {
                kind: header.kind,
                indent: header.indent,
                range: header.range_start..range_end,
                items,
            });
            index = body_end;
        } else {
            index += 1;
        }
    }

    sections
}

fn blank_line_continues_section(
    style: Style,
    previous_lines: &[Line<'_>],
    lines: &[Line<'_>],
    header: Header,
) -> bool {
    let Some((offset, non_blank_line)) = lines
        .iter()
        .enumerate()
        .find(|(_, line)| !line.text.trim().is_empty())
    else {
        return false;
    };

    if parse_section_header(style, lines, offset).is_some_and(|next| next.indent <= header.indent) {
        return false;
    }

    match style {
        Style::Google => line_belongs_to_google_body_after_blank(header, non_blank_line.text),
        Style::Numpy => {
            let line_indent = indentation(non_blank_line.text);
            if line_indent > header.indent {
                return true;
            }

            line_indent == header.indent
                && match header.kind {
                    SectionKind::Parameters | SectionKind::Attributes => {
                        numpy_named_item_starts(non_blank_line, &lines[offset + 1..])
                    }
                    SectionKind::Returns => {
                        numpy_return_item_starts(non_blank_line, previous_lines)
                    }
                    SectionKind::Raises => {
                        numpy_raise_item_starts(non_blank_line, &lines[offset + 1..])
                    }
                }
        }
    }
}

fn line_belongs_to_body(
    style: Style,
    line: &Line<'_>,
    previous_lines: &[Line<'_>],
    following_lines: &[Line<'_>],
    header: Header,
) -> bool {
    match style {
        Style::Google => line_belongs_to_google_body(header, line.text),
        Style::Numpy => line_belongs_to_numpy_body(header, line, previous_lines, following_lines),
    }
}

fn line_belongs_to_numpy_body(
    header: Header,
    line: &Line<'_>,
    previous_lines: &[Line<'_>],
    following_lines: &[Line<'_>],
) -> bool {
    let line_indent = indentation(line.text);
    if line_indent > header.indent {
        return true;
    }

    if line_indent != header.indent {
        return false;
    }

    match header.kind {
        SectionKind::Parameters | SectionKind::Attributes => {
            numpy_named_item_starts(line, following_lines)
        }
        SectionKind::Raises => numpy_raise_item_starts(line, following_lines),
        SectionKind::Returns => numpy_return_item_starts(line, previous_lines),
    }
}

fn line_belongs_to_google_body(header: Header, line: &str) -> bool {
    let line_indent = indentation(line);
    line_indent > header.indent
        // `documentation_trim` ignores the first docstring line when computing common
        // indentation, so the body of a first-line section can be dedented to the
        // header's column before structured parsing sees it.
        || (header.range_start == 0 && line_indent == header.indent)
}

fn line_belongs_to_google_body_after_blank(header: Header, line: &str) -> bool {
    if !line_belongs_to_google_body(header, line) {
        return false;
    }

    let line_indent = indentation(line);
    if header.range_start == 0
        && line_indent == header.indent
        && is_google_section_like_header(line.trim())
    {
        return false;
    }

    if header.range_start != 0 || line_indent > header.indent {
        return true;
    }

    match header.kind {
        SectionKind::Parameters | SectionKind::Attributes | SectionKind::Raises => {
            parse_google_named_item(line.trim()).is_some()
        }
        SectionKind::Returns => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    kind: SectionKind,
    indent: usize,
    body_start: usize,
    range_start: usize,
    range_end: usize,
}

fn parse_section_header(style: Style, lines: &[Line<'_>], index: usize) -> Option<Header> {
    match style {
        Style::Google => {
            let line = lines.get(index)?;
            let (kind, indent) = parse_google_header(line.text)?;
            Some(Header {
                kind,
                indent,
                body_start: index + 1,
                range_start: line.start,
                range_end: line.end,
            })
        }
        Style::Numpy => {
            let line = lines.get(index)?;
            let underline = lines.get(index + 1)?;
            if !is_numpy_underline(underline.text) {
                return None;
            }

            let kind = parse_numpy_header(line.text)?;
            Some(Header {
                kind,
                indent: indentation(line.text),
                body_start: index + 2,
                range_start: line.start,
                range_end: underline.end,
            })
        }
    }
}

fn parse_google_header(line: &str) -> Option<(SectionKind, usize)> {
    let name = line.trim().strip_suffix(':')?.trim();
    let kind = match normalized_google_section_name(name).as_str() {
        "args" | "arguments" | "parameters" | "keyword args" | "keyword arguments" => {
            SectionKind::Parameters
        }
        "attributes" => SectionKind::Attributes,
        "returns" | "return" => SectionKind::Returns,
        "raises" | "raise" => SectionKind::Raises,
        _ => return None,
    };
    Some((kind, indentation(line)))
}

fn is_google_section_like_header(line: &str) -> bool {
    let Some(name) = line.strip_suffix(':') else {
        return false;
    };

    matches!(
        normalized_google_section_name(name).as_str(),
        "args"
            | "arguments"
            | "parameters"
            | "keyword args"
            | "keyword arguments"
            | "attributes"
            | "returns"
            | "return"
            | "raises"
            | "raise"
            | "example"
            | "examples"
            | "note"
            | "notes"
            | "other parameters"
            | "references"
            | "see also"
            | "todo"
            | "todos"
            | "warning"
            | "warnings"
            | "yield"
            | "yields"
    )
}

fn normalized_google_section_name(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn parse_numpy_header(line: &str) -> Option<SectionKind> {
    match line.trim().to_ascii_lowercase().as_str() {
        "parameters" => Some(SectionKind::Parameters),
        "attributes" => Some(SectionKind::Attributes),
        "returns" | "return" => Some(SectionKind::Returns),
        "raises" | "raise" => Some(SectionKind::Raises),
        _ => None,
    }
}

fn is_numpy_underline(line: &str) -> bool {
    let line = line.trim();
    line.len() >= 3 && line.chars().all(|char| char == '-')
}

fn numpy_named_item_starts(line: &Line<'_>, following_lines: &[Line<'_>]) -> bool {
    let trimmed = line.text.trim();
    if has_numpy_type_separator(trimmed) {
        return true;
    }

    numpy_untyped_item_starts(trimmed, line, following_lines)
}

fn numpy_untyped_item_starts(trimmed: &str, line: &Line<'_>, following_lines: &[Line<'_>]) -> bool {
    is_numpy_item_name(trimmed)
        && following_lines
            .iter()
            .find(|line| !line.text.trim().is_empty())
            .is_some_and(|next| indentation(next.text) > indentation(line.text))
}

fn has_numpy_type_separator(line: &str) -> bool {
    split_numpy_type_separator(line).is_some()
}

fn split_numpy_type_separator(line: &str) -> Option<(&str, &str)> {
    let (name, ty) = split_once_unbracketed_colon(line)?;
    if !name.chars().last().is_some_and(char::is_whitespace) {
        return None;
    }

    let name = name.trim();
    let ty = ty.trim();
    if !is_numpy_item_name(name) || ty.is_empty() {
        return None;
    }

    Some((name, ty))
}

fn is_numpy_item_name(name: &str) -> bool {
    name.split(',').all(|part| {
        let part = part.trim();
        let part = part
            .strip_prefix("**")
            .or_else(|| part.strip_prefix('*'))
            .unwrap_or(part);

        !part.is_empty() && part.split('.').all(is_numpy_name_part)
    })
}

fn is_numpy_name_part(part: &str) -> bool {
    let mut chars = part.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|char| char == '_' || char.is_ascii_alphanumeric())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionKind {
    Parameters,
    Attributes,
    Returns,
    Raises,
}

impl SectionKind {
    fn docstring_section_kind(self) -> DocstringSectionKind {
        match self {
            Self::Parameters => DocstringSectionKind::Parameters,
            Self::Attributes => DocstringSectionKind::Attributes,
            Self::Returns => DocstringSectionKind::Returns,
            Self::Raises => DocstringSectionKind::Raises,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Item {
    display_name: Option<String>,
    ty: Option<String>,
    description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemBuilder {
    display_name: Option<String>,
    ty: Option<String>,
    description_lines: Vec<DescriptionLine>,
}

impl ItemBuilder {
    fn finish(self) -> Item {
        Item {
            display_name: self.display_name,
            ty: self.ty,
            description: normalize_description(self.description_lines),
        }
    }

    fn push_description(&mut self, line: &str) {
        self.description_lines
            .push(DescriptionLine::Source(line.to_string()));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DescriptionLine {
    Normalized(String),
    Source(String),
}

impl DescriptionLine {
    fn normalized(line: &str) -> Self {
        Self::Normalized(line.trim().to_string())
    }
}

fn parse_items(style: Style, kind: SectionKind, body: &[Line<'_>]) -> Option<Vec<Item>> {
    match style {
        Style::Google => parse_google_items(kind, body),
        Style::Numpy => parse_numpy_items(kind, body),
    }
}

fn parse_google_items(kind: SectionKind, body: &[Line<'_>]) -> Option<Vec<Item>> {
    if kind == SectionKind::Returns {
        return parse_google_return_item(body);
    }

    parse_named_items(body, parse_google_named_item)
}

fn parse_google_named_item(line: &str) -> Option<ItemBuilder> {
    let (name, description) = split_once_unbracketed_colon(line)?;
    let (name, ty) = parse_parenthesized_type(name.trim());
    if name.is_empty() {
        return None;
    }

    Some(ItemBuilder {
        display_name: Some(name.to_string()),
        ty: ty.map(str::to_string),
        description_lines: vec![DescriptionLine::normalized(description)],
    })
}

fn split_once_unbracketed_colon(line: &str) -> Option<(&str, &str)> {
    let mut parentheses = 0usize;
    let mut brackets = 0usize;
    let mut braces = 0usize;
    let mut quote = None;
    let mut escaped = false;

    for (index, char) in line.char_indices() {
        if let Some(quote_char) = quote {
            if escaped {
                escaped = false;
            } else if char == '\\' {
                escaped = true;
            } else if char == quote_char {
                quote = None;
            }
            continue;
        }

        match char {
            '\'' | '"' => quote = Some(char),
            '(' => parentheses += 1,
            ')' => parentheses = parentheses.saturating_sub(1),
            '[' => brackets += 1,
            ']' => brackets = brackets.saturating_sub(1),
            '{' => braces += 1,
            '}' => braces = braces.saturating_sub(1),
            ':' if parentheses == 0 && brackets == 0 && braces == 0 => {
                return Some((&line[..index], &line[index + ':'.len_utf8()..]));
            }
            _ => {}
        }
    }

    None
}

fn parse_numpy_items(kind: SectionKind, body: &[Line<'_>]) -> Option<Vec<Item>> {
    match kind {
        SectionKind::Parameters | SectionKind::Attributes => {
            parse_named_items(body, parse_numpy_named_item)
        }
        SectionKind::Returns => parse_numpy_return_items(body),
        SectionKind::Raises => parse_numpy_raise_items(body),
    }
}

fn parse_named_items(
    body: &[Line<'_>],
    parse_item: fn(&str) -> Option<ItemBuilder>,
) -> Option<Vec<Item>> {
    let mut items = Vec::new();
    let mut current: Option<ItemBuilder> = None;
    let mut item_indent = None;

    for line in body {
        let trimmed = line.text.trim();
        if trimmed.is_empty() {
            if let Some(current) = &mut current {
                current.push_description("");
            }
            continue;
        }

        let starts_item = item_indent.is_none_or(|indent| indentation(line.text) == indent);
        if starts_item && let Some(item) = parse_item(trimmed) {
            if let Some(current) = current.replace(item) {
                items.push(current.finish());
            }
            item_indent.get_or_insert_with(|| indentation(line.text));
            continue;
        }

        let current = current.as_mut()?;
        current.push_description(line.text);
    }

    if let Some(current) = current {
        items.push(current.finish());
    }
    (!items.is_empty()).then_some(items)
}

fn parse_numpy_named_item(line: &str) -> Option<ItemBuilder> {
    let (name, ty) = split_numpy_type_separator(line).map_or_else(
        || is_numpy_item_name(line.trim()).then_some((line.trim(), None)),
        |(name, ty)| Some((name, Some(ty))),
    )?;

    Some(ItemBuilder {
        display_name: Some(name.to_string()),
        ty: ty.filter(|ty| !ty.is_empty()).map(str::to_string),
        description_lines: Vec::new(),
    })
}

fn parse_numpy_return_items(body: &[Line<'_>]) -> Option<Vec<Item>> {
    if body
        .iter()
        .map(|line| line.text.trim())
        .find(|line| !line.is_empty())
        .is_some_and(|line| parse_numpy_named_return_item(line).is_some())
    {
        return parse_named_items(body, parse_numpy_named_return_item);
    }

    let mut lines = body.iter().skip_while(|line| line.text.trim().is_empty());
    let first = lines.next()?;
    let first = first.text.trim();

    Some(vec![Item {
        display_name: None,
        ty: Some(first.to_string()),
        description: normalize_description(
            lines
                .map(|line| DescriptionLine::Source(line.text.to_string()))
                .collect(),
        ),
    }])
}

fn numpy_return_item_starts(line: &Line<'_>, previous_lines: &[Line<'_>]) -> bool {
    let trimmed = line.text.trim();
    parse_numpy_named_return_item(trimmed).is_some()
        || (!previous_lines
            .iter()
            .any(|line| !line.text.trim().is_empty())
            && is_numpy_anonymous_return_type(trimmed))
}

fn is_numpy_anonymous_return_type(line: &str) -> bool {
    !line.is_empty() && !line.ends_with('.') && !line.ends_with(':')
}

fn parse_numpy_named_return_item(line: &str) -> Option<ItemBuilder> {
    let (name, ty) = split_numpy_type_separator(line)?;

    Some(ItemBuilder {
        display_name: Some(name.to_string()),
        ty: Some(ty.to_string()),
        description_lines: Vec::new(),
    })
}

fn parse_numpy_raise_items(body: &[Line<'_>]) -> Option<Vec<Item>> {
    parse_named_items(body, parse_numpy_raise_item)
}

fn numpy_raise_item_starts(line: &Line<'_>, following_lines: &[Line<'_>]) -> bool {
    let trimmed = line.text.trim();
    parse_numpy_raise_item(trimmed).is_some_and(|item| !item.description_lines.is_empty())
        || numpy_untyped_item_starts(trimmed, line, following_lines)
}

fn parse_numpy_raise_item(line: &str) -> Option<ItemBuilder> {
    let (name, description) = line
        .split_once(':')
        .map_or((line.trim(), None), |(name, description)| {
            (name.trim(), Some(description.trim()))
        });
    if !is_numpy_item_name(name) {
        return None;
    }

    Some(ItemBuilder {
        display_name: Some(name.to_string()),
        ty: None,
        description_lines: description
            .filter(|description| !description.is_empty())
            .map(DescriptionLine::normalized)
            .into_iter()
            .collect(),
    })
}

fn parse_google_return_item(body: &[Line<'_>]) -> Option<Vec<Item>> {
    let mut lines = body.iter().skip_while(|line| line.text.trim().is_empty());
    let first = lines.next()?;
    let first = first.text.trim();
    let (ty, first_description) = split_once_unbracketed_colon(first)
        .map_or((None, first), |(ty, description)| {
            (Some(ty.trim()), description.trim())
        });

    let mut description_lines = vec![DescriptionLine::normalized(first_description)];
    description_lines.extend(lines.map(|line| DescriptionLine::Source(line.text.to_string())));
    Some(vec![Item {
        display_name: None,
        ty: ty.map(str::to_string),
        description: normalize_description(description_lines),
    }])
}

fn parse_parenthesized_type(name: &str) -> (&str, Option<&str>) {
    if !name.ends_with(')') {
        return (name, None);
    }

    let mut depth = 0usize;
    for (index, char) in name.char_indices().rev() {
        match char {
            ')' => depth += 1,
            '(' => {
                if depth == 0 {
                    return (name, None);
                }
                depth -= 1;
                if depth == 0 {
                    let display_name = name[..index].trim();
                    let ty = name[index + '('.len_utf8()..name.len() - ')'.len_utf8()].trim();
                    return if display_name.is_empty() || ty.is_empty() {
                        (name, None)
                    } else {
                        (display_name, Some(ty))
                    };
                }
            }
            _ => {}
        }
    }

    (name, None)
}

fn normalize_description(lines: Vec<DescriptionLine>) -> String {
    let dedent = lines
        .iter()
        .filter_map(|line| match line {
            DescriptionLine::Source(line) if !line.trim().is_empty() => Some(indentation(line)),
            DescriptionLine::Normalized(_) | DescriptionLine::Source(_) => None,
        })
        .min()
        .unwrap_or(0);

    let mut description = lines
        .into_iter()
        .map(|line| match line {
            DescriptionLine::Normalized(line) => line,
            DescriptionLine::Source(line) => {
                strip_indentation(&line, dedent).trim_end().to_string()
            }
        })
        .skip_while(String::is_empty)
        .collect::<Vec<_>>();
    while description.last().is_some_and(String::is_empty) {
        description.pop();
    }
    description.join("\n")
}

fn strip_indentation(line: &str, width: usize) -> &str {
    let mut indentation_width = 0;
    for (index, char) in line.char_indices() {
        let char_width = match char {
            ' ' => 1,
            '\t' => 8,
            _ => return &line[index..],
        };

        if indentation_width + char_width > width {
            return &line[index..];
        }

        indentation_width += char_width;
        if indentation_width == width {
            return &line[index + char.len_utf8()..];
        }
    }

    ""
}

fn indentation(line: &str) -> usize {
    leading_indentation(line)
        .chars()
        .map(|char| if char == '\t' { 8 } else { 1 })
        .sum()
}
