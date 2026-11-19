/// Incremental byte-buffered line parser for Server-Sent Events (SSE).
///
/// Splits strictly on `0x0A` (`\n`) before UTF-8 decoding so multi-byte
/// characters split across network chunks never produce replacement characters.
#[derive(Debug, Default)]
pub struct SseLineParser {
    buf: Vec<u8>,
}

impl SseLineParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes from an HTTP chunk and return all complete lines.
    pub fn push_chunk(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut lines = Vec::new();
        let mut start = 0;

        while let Some(rel_pos) = self.buf[start..].iter().position(|&b| b == b'\n') {
            let end = start + rel_pos;
            let mut slice = &self.buf[start..end];
            if slice.last() == Some(&b'\r') {
                slice = &slice[..slice.len() - 1];
            }
            lines.push(String::from_utf8_lossy(slice).into_owned());
            start = end + 1;
        }

        if start > 0 {
            self.buf.drain(..start);
        }
        lines
    }

    /// Flush any trailing incomplete line on stream EOF.
    pub fn finish(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            return None;
        }
        let mut slice = self.buf.as_slice();
        if slice.last() == Some(&b'\r') {
            slice = &slice[..slice.len() - 1];
        }
        let line = String::from_utf8_lossy(slice).into_owned();
        self.buf.clear();
        Some(line)
    }
}

/// Extract the trimmed payload of an SSE `data: ...` line, if present.
pub fn extract_sse_data(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed.strip_prefix("data:").map(str::trim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_multibyte_utf8_split_across_chunks() {
        let mut parser = SseLineParser::new();
        let line = "data: {\"text\":\"héllo 🚀 world\"}\n";
        let bytes = line.as_bytes();
        // Split right inside the 4-byte rocket emoji
        let split_at = bytes.iter().position(|&b| b == 0xF0).unwrap() + 2;

        let first = parser.push_chunk(&bytes[..split_at]);
        assert!(first.is_empty());

        let second = parser.push_chunk(&bytes[split_at..]);
        assert_eq!(second, vec![r#"data: {"text":"héllo 🚀 world"}"#]);
        assert!(parser.finish().is_none());
    }

    #[test]
    fn flushes_trailing_line_without_newline_at_eof() {
        let mut parser = SseLineParser::new();
        assert!(parser.push_chunk(b"data: [DONE]").is_empty());
        assert_eq!(parser.finish().as_deref(), Some("data: [DONE]"));
        assert!(parser.finish().is_none());
    }
}
