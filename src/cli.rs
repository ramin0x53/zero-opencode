use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// CLI arguments for zero-opencode.
///
/// Configuration is provided via a YAML file (`-c` / `--config`) rather than
/// individual CLI flags.
#[derive(Parser, Debug)]
#[command(
    name = "zero-opencode",
    version,
    about = "OpenCode Free to OpenAI-compatible API proxy"
)]
pub struct Cli {
    /// Path to YAML configuration file (defaults to ./config.yaml if present, else built-in defaults)
    #[arg(short = 'c', long = "config", global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Supported CLI subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the OpenAI-compatible HTTP server (default when no subcommand is given)
    Serve,
    /// Fetch and print all currently available OpenCode free models
    Models {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Run a live smoke test (streaming + non-streaming) against OpenCode Free
    Test {
        /// Model ID to test (defaults to the first preferred active free model)
        model: Option<String>,
    },
}
