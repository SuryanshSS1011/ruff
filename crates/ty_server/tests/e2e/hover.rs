use anyhow::Result;
use lsp_types::{FileChangeType, FileEvent, Position, TextDocumentContentChangeEvent};
use ruff_db::system::SystemPath;

use crate::TestServerBuilder;

#[test]
fn hover_after_external_file_change() -> Result<()> {
    let workspace_root = SystemPath::new("src");
    let foo = SystemPath::new("src/test.py");
    let initial_content = "\
def test(a: str) -> str:
    return \"42\"
";
    let changed_content = "\
from typing import Literal

def test(a: Literal[\"42\"]) -> str:
    return \"42\"
";

    let mut server = TestServerBuilder::new()?
        .with_workspace(workspace_root, None)?
        .with_file(foo, initial_content)?
        .build()
        .wait_until_workspaces_are_initialized();

    server.open_text_document(foo, initial_content, 1);
    server.write_file(foo, changed_content)?;
    server.did_change_watched_files(vec![FileEvent {
        uri: server.file_uri(foo),
        typ: FileChangeType::CHANGED,
    }]);
    server.change_text_document(
        foo,
        vec![TextDocumentContentChangeEvent {
            range: None,
            range_length: None,
            text: changed_content.to_string(),
        }],
        2,
    );

    let hover = server.hover_request(foo, Position::new(2, 13));

    assert!(hover.is_some());

    Ok(())
}
