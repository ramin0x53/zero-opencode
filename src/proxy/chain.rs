use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{Context, bail};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio_socks::tcp::{Socks4Stream, Socks5Stream};
use tracing::{debug, warn};
use wreq::Proxy;

use crate::proxy::endpoint::{ProxyAuth, ProxyEndpoint, ProxyId, ProxyProtocol};

const MAX_CONNECT_HEADER_BYTES: usize = 8192;
const LOCAL_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Object-safe async stream trait used for composable multi-hop proxy dialing.
pub trait AsyncStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> AsyncStream for T {}

pub type BoxedStream = Pin<Box<dyn AsyncStream>>;

/// Stream wrapper preserving bytes already read beyond an HTTP `CONNECT` header boundary.
struct PrefixedStream<S> {
    prefix: Vec<u8>,
    pos: usize,
    inner: S,
}

impl<S> PrefixedStream<S> {
    fn new(prefix: Vec<u8>, inner: S) -> Self {
        Self {
            prefix,
            pos: 0,
            inner,
        }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for PrefixedStream<S> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.pos < self.prefix.len() {
            let remaining = self.prefix.len() - self.pos;
            let to_copy = remaining.min(buf.remaining());
            let end = self.pos + to_copy;
            buf.put_slice(&self.prefix[self.pos..end]);
            self.pos = end;
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for PrefixedStream<S> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

/// RFC 4648 standard Base64 encoder for HTTP `Proxy-Authorization: Basic`.
pub fn encode_base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn format_authority(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

#[derive(Debug, thiserror::Error)]
enum DialError {
    #[error("proxy transport/auth error: {0}")]
    ProxyBroken(#[source] anyhow::Error),
    #[error("target unreachable via proxy: {0}")]
    TargetUnreachable(#[source] anyhow::Error),
}

async fn dial_http_connect<S>(
    mut stream: S,
    auth: Option<&ProxyAuth>,
    target_host: &str,
    target_port: u16,
) -> Result<BoxedStream, DialError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let authority = format_authority(target_host, target_port);
    let mut req = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n");
    if let Some(creds) = auth {
        let raw = format!("{}:{}", creds.username, creds.password);
        let encoded = encode_base64(raw.as_bytes());
        req.push_str(&format!("Proxy-Authorization: Basic {encoded}\r\n"));
    }
    req.push_str("\r\n");

    stream
        .write_all(req.as_bytes())
        .await
        .map_err(|e| DialError::ProxyBroken(e.into()))?;
    stream
        .flush()
        .await
        .map_err(|e| DialError::ProxyBroken(e.into()))?;

    let mut buf = Vec::with_capacity(512);
    let mut tmp = [0u8; 512];
    let header_end = loop {
        let n = stream
            .read(&mut tmp)
            .await
            .map_err(|e| DialError::ProxyBroken(e.into()))?;
        if n == 0 {
            return Err(DialError::ProxyBroken(anyhow::anyhow!(
                "HTTP CONNECT tunnel closed unexpectedly (EOF)"
            )));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > MAX_CONNECT_HEADER_BYTES {
            return Err(DialError::ProxyBroken(anyhow::anyhow!(
                "HTTP CONNECT response header exceeded {MAX_CONNECT_HEADER_BYTES} bytes"
            )));
        }
    };

    let header_text = String::from_utf8_lossy(&buf[..header_end]);
    let status_line = header_text.lines().next().unwrap_or("").trim();
    let code = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| {
            DialError::ProxyBroken(anyhow::anyhow!(
                "malformed HTTP CONNECT status line: `{status_line}`"
            ))
        })?;

    if !(200..300).contains(&code) {
        let err = anyhow::anyhow!("HTTP CONNECT rejected with status {code} (`{status_line}`)");
        if matches!(code, 502..=504) {
            return Err(DialError::TargetUnreachable(err));
        }
        return Err(DialError::ProxyBroken(err));
    }

    let leftover = buf.split_off(header_end);
    if leftover.is_empty() {
        Ok(Box::pin(stream))
    } else {
        Ok(Box::pin(PrefixedStream::new(leftover, stream)))
    }
}

fn classify_socks_error(err: tokio_socks::Error, context_msg: &'static str) -> DialError {
    use tokio_socks::Error as SocksErr;
    match err {
        SocksErr::GeneralSocksServerFailure
        | SocksErr::ConnectionNotAllowedByRuleset
        | SocksErr::NetworkUnreachable
        | SocksErr::HostUnreachable
        | SocksErr::ConnectionRefused
        | SocksErr::TtlExpired => {
            DialError::TargetUnreachable(anyhow::Error::new(err).context(context_msg))
        }
        other => DialError::ProxyBroken(anyhow::Error::new(other).context(context_msg)),
    }
}

/// Establish a tunnel to `(target_host, target_port)` over an already-connected `stream`
/// speaking `proxy.protocol` (`http`, `socks4`, `socks4a`, `socks5`, `socks5h`).
async fn dial_through_proxy_classified<S>(
    stream: S,
    proxy: &ProxyEndpoint,
    target_host: &str,
    target_port: u16,
) -> Result<BoxedStream, DialError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    match proxy.protocol.to_remote_dns() {
        ProxyProtocol::Socks5h | ProxyProtocol::Socks5 => {
            let tunneled = match &proxy.auth {
                Some(auth) if !auth.username.is_empty() && !auth.password.is_empty() => {
                    Socks5Stream::connect_with_password_and_socket(
                        stream,
                        (target_host, target_port),
                        &auth.username,
                        &auth.password,
                    )
                    .await
                    .map_err(|e| classify_socks_error(e, "SOCKS5 authenticated handshake failed"))?
                    .into_inner()
                }
                _ => Socks5Stream::connect_with_socket(stream, (target_host, target_port))
                    .await
                    .map_err(|e| classify_socks_error(e, "SOCKS5 handshake failed"))?
                    .into_inner(),
            };
            Ok(Box::pin(tunneled))
        }
        ProxyProtocol::Socks4a | ProxyProtocol::Socks4 => {
            let tunneled = match &proxy.auth {
                Some(auth) if !auth.username.is_empty() => {
                    Socks4Stream::connect_with_userid_and_socket(
                        stream,
                        (target_host, target_port),
                        &auth.username,
                    )
                    .await
                    .map_err(|e| classify_socks_error(e, "SOCKS4a userid handshake failed"))?
                    .into_inner()
                }
                _ => Socks4Stream::connect_with_socket(stream, (target_host, target_port))
                    .await
                    .map_err(|e| classify_socks_error(e, "SOCKS4a handshake failed"))?
                    .into_inner(),
            };
            Ok(Box::pin(tunneled))
        }
        ProxyProtocol::Http => {
            dial_http_connect(stream, proxy.auth.as_ref(), target_host, target_port).await
        }
        ProxyProtocol::Https => Err(DialError::ProxyBroken(anyhow::anyhow!(
            "chaining through HTTPS-TLS proxy `{}` is not supported; use http://, socks4://, or socks5://",
            proxy.redacted()
        ))),
    }
}

/// Public helper wrapping `dial_through_proxy_classified` into `anyhow::Result`.
pub async fn dial_through_proxy<S>(
    stream: S,
    proxy: &ProxyEndpoint,
    target_host: &str,
    target_port: u16,
) -> anyhow::Result<BoxedStream>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    dial_through_proxy_classified(stream, proxy, target_host, target_port)
        .await
        .map_err(anyhow::Error::from)
}

/// In-process authenticated loopback SOCKS5h bridge that chains every exit proxy
/// through a configured base/chain proxy (`opencode.proxy`).
pub struct ChainBridge {
    base_proxy: ProxyEndpoint,
    local_addr: SocketAddr,
    secret: String,
    endpoints: Arc<RwLock<HashMap<ProxyId, ProxyEndpoint>>>,
    connect_timeout: Duration,
    hop1_failing: Arc<AtomicBool>,
    last_hop1_error: Arc<Mutex<Option<String>>>,
    shutdown_tx: watch::Sender<bool>,
}

impl ChainBridge {
    /// Start the loopback SOCKS5h bridge chaining through `base_proxy`.
    pub async fn start(
        base_proxy: ProxyEndpoint,
        connect_timeout: Duration,
    ) -> anyhow::Result<Arc<Self>> {
        if base_proxy.protocol == ProxyProtocol::Https {
            bail!(
                "opencode.proxy (`{}`) uses https:// which cannot be chained; use http://, socks4://, or socks5://",
                base_proxy.redacted()
            );
        }

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .context("failed to bind ChainBridge loopback listener")?;
        let local_addr = listener.local_addr()?;

        let secret_bytes: [u8; 16] = rand::random();
        let secret = hex::encode(secret_bytes);
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        let bridge = Arc::new(Self {
            base_proxy,
            local_addr,
            secret,
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            connect_timeout,
            hop1_failing: Arc::new(AtomicBool::new(false)),
            last_hop1_error: Arc::new(Mutex::new(None)),
            shutdown_tx,
        });

        let accept_bridge = Arc::clone(&bridge);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    accepted = listener.accept() => {
                        match accepted {
                            Ok((stream, _peer)) => {
                                let conn_bridge = Arc::clone(&accept_bridge);
                                tokio::spawn(async move {
                                    if let Err(err) = conn_bridge.handle_client(stream).await {
                                        debug!(error = %err, "ChainBridge connection ended");
                                    }
                                });
                            }
                            Err(err) => {
                                warn!(error = %err, "ChainBridge accept error");
                            }
                        }
                    }
                }
            }
        });

        Ok(bridge)
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn chained_via(&self) -> String {
        self.base_proxy.redacted()
    }

    pub fn is_hop1_failing(&self) -> bool {
        self.hop1_failing.load(Ordering::Relaxed)
    }

    /// Actively probe TCP reachability to the Hop 1 chain proxy (`opencode.proxy`),
    /// clearing `hop1_failing` once reachable so probing and traffic can resume.
    pub async fn check_hop1_reachable(&self) -> bool {
        let timeout = self.connect_timeout.min(Duration::from_secs(3));
        let ok = matches!(
            tokio::time::timeout(
                timeout,
                TcpStream::connect((self.base_proxy.host.as_str(), self.base_proxy.port)),
            )
            .await,
            Ok(Ok(_))
        );
        self.hop1_failing.store(!ok, Ordering::Relaxed);
        ok
    }

    pub fn last_hop1_error(&self) -> Option<String> {
        self.last_hop1_error.lock().ok().and_then(|g| g.clone())
    }

    /// Replace the registered exit proxy map atomically (in-flight tunnels keep running).
    pub fn replace_endpoints(&self, eps: impl IntoIterator<Item = ProxyEndpoint>) {
        let mut guard = self
            .endpoints
            .write()
            .expect("ChainBridge endpoints write lock");
        guard.clear();
        for ep in eps {
            guard.insert(ep.id, ep);
        }
    }

    /// Register or update a single exit proxy (useful for tests or ad-hoc probes).
    pub fn register_endpoint(&self, ep: ProxyEndpoint) {
        let mut guard = self
            .endpoints
            .write()
            .expect("ChainBridge endpoints write lock");
        guard.insert(ep.id, ep);
    }

    /// Build a `wreq::Proxy` targeting this bridge with per-proxy RFC 1929 credentials.
    pub fn wreq_proxy_for(&self, id: ProxyId) -> anyhow::Result<Proxy> {
        let url = format!(
            "socks5h://{}:{}@127.0.0.1:{}",
            id.0,
            self.secret,
            self.local_addr.port()
        );
        Proxy::all(&url).context("failed to construct ChainBridge wreq::Proxy")
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    async fn handle_client(&self, mut client: TcpStream) -> anyhow::Result<()> {
        let _ = client.set_nodelay(true);

        // 1. SOCKS5 greeting + RFC 1929 auth + target request (bounded local timeout)
        let (exit_proxy, target_host, target_port) =
            tokio::time::timeout(LOCAL_HANDSHAKE_TIMEOUT, self.read_socks5_request(&mut client))
                .await
                .context("ChainBridge local SOCKS5 handshake timed out")??;

        // 2. Hop 1: connect through base_proxy -> exit_proxy
        let base = &self.base_proxy;
        let tcp = match tokio::time::timeout(
            self.connect_timeout,
            TcpStream::connect((base.host.as_str(), base.port)),
        )
        .await
        {
            Ok(Ok(tcp)) => {
                let _ = tcp.set_nodelay(true);
                tcp
            }
            Ok(Err(err)) => {
                let msg = format!("failed TCP connect to chain proxy {}: {err}", base.redacted());
                self.mark_hop1_error(msg.clone());
                let _ = send_socks5_reply(&mut client, 0x03).await;
                bail!("{msg}");
            }
            Err(_) => {
                let msg = format!("TCP connect to chain proxy {} timed out", base.redacted());
                self.mark_hop1_error(msg.clone());
                let _ = send_socks5_reply(&mut client, 0x03).await;
                bail!("{msg}");
            }
        };

        let hop1_res = tokio::time::timeout(
            self.connect_timeout,
            dial_through_proxy_classified(tcp, base, &exit_proxy.host, exit_proxy.port),
        )
        .await;

        let hop1_stream = match hop1_res {
            Ok(Ok(stream)) => {
                self.hop1_failing.store(false, Ordering::Relaxed);
                stream
            }
            Ok(Err(DialError::TargetUnreachable(err))) => {
                // Chain proxy itself responded validly, exit proxy is unreachable.
                self.hop1_failing.store(false, Ordering::Relaxed);
                let _ = send_socks5_reply(&mut client, 0x04).await;
                return Err(err.context(format!(
                    "exit proxy {} unreachable through {}",
                    exit_proxy.redacted(),
                    base.redacted()
                )));
            }
            Ok(Err(DialError::ProxyBroken(err))) => {
                let wrapped = err.context(format!("chain proxy {} error", base.redacted()));
                self.mark_hop1_error(wrapped.to_string());
                let _ = send_socks5_reply(&mut client, 0x03).await;
                return Err(wrapped);
            }
            Err(_) => {
                // TCP to chain proxy succeeded; timeout occurred waiting for exit proxy.
                self.hop1_failing.store(false, Ordering::Relaxed);
                let _ = send_socks5_reply(&mut client, 0x04).await;
                bail!(
                    "timed out connecting to exit proxy {} through {}",
                    exit_proxy.redacted(),
                    base.redacted()
                );
            }
        };

        // 3. Hop 2: dial through exit_proxy -> target_host:target_port
        let hop2_res = tokio::time::timeout(
            self.connect_timeout,
            dial_through_proxy_classified(hop1_stream, &exit_proxy, &target_host, target_port),
        )
        .await;

        let mut hop2_stream = match hop2_res {
            Ok(Ok(stream)) => stream,
            Ok(Err(err)) => {
                let _ = send_socks5_reply(&mut client, 0x05).await;
                return Err(anyhow::Error::from(err));
            }
            Err(_) => {
                let _ = send_socks5_reply(&mut client, 0x04).await;
                bail!(
                    "exit proxy {} timed out dialing {target_host}:{target_port}",
                    exit_proxy.redacted()
                );
            }
        };

        // 4. Reply success & relay bidirectional traffic
        send_socks5_reply(&mut client, 0x00).await?;
        let _ = tokio::io::copy_bidirectional(&mut client, &mut hop2_stream).await;
        Ok(())
    }

    fn mark_hop1_error(&self, msg: String) {
        self.hop1_failing.store(true, Ordering::Relaxed);
        if let Ok(mut guard) = self.last_hop1_error.lock() {
            *guard = Some(msg);
        }
    }

    async fn read_socks5_request(
        &self,
        client: &mut TcpStream,
    ) -> anyhow::Result<(ProxyEndpoint, String, u16)> {
        // Greeting: VER, NMETHODS
        let mut hdr = [0u8; 2];
        client.read_exact(&mut hdr).await?;
        if hdr[0] != 0x05 || hdr[1] == 0 {
            bail!("unsupported SOCKS version or empty methods");
        }
        let mut methods = vec![0u8; hdr[1] as usize];
        client.read_exact(&mut methods).await?;
        if !methods.contains(&0x02) {
            client.write_all(&[0x05, 0xFF]).await?;
            bail!("ChainBridge requires RFC 1929 username/password authentication");
        }
        client.write_all(&[0x05, 0x02]).await?;

        // RFC 1929 auth: VER=0x01, ULEN, UNAME, PLEN, PASSWD
        let mut auth_hdr = [0u8; 2];
        client.read_exact(&mut auth_hdr).await?;
        if auth_hdr[0] != 0x01 {
            let _ = client.write_all(&[0x01, 0x01]).await;
            bail!("invalid RFC 1929 auth version");
        }
        let mut uname = vec![0u8; auth_hdr[1] as usize];
        client.read_exact(&mut uname).await?;
        let plen = client.read_u8().await? as usize;
        let mut passwd = vec![0u8; plen];
        client.read_exact(&mut passwd).await?;

        let secret_ok = constant_time_eq(&passwd, self.secret.as_bytes());
        let proxy_id = if secret_ok {
            std::str::from_utf8(&uname)
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(ProxyId)
        } else {
            None
        };

        let Some(proxy_id) = proxy_id else {
            let _ = client.write_all(&[0x01, 0x01]).await;
            bail!("ChainBridge authentication failed");
        };
        client.write_all(&[0x01, 0x00]).await?;

        // Request: VER, CMD, RSV, ATYP
        let mut req_hdr = [0u8; 4];
        client.read_exact(&mut req_hdr).await?;
        if req_hdr[0] != 0x05 {
            bail!("invalid SOCKS5 request version");
        }
        if req_hdr[1] != 0x01 {
            let _ = send_socks5_reply(client, 0x07).await;
            bail!("unsupported SOCKS5 command {}", req_hdr[1]);
        }

        let target_host = match req_hdr[3] {
            0x01 => {
                let mut ip = [0u8; 4];
                client.read_exact(&mut ip).await?;
                IpAddr::V4(Ipv4Addr::from(ip)).to_string()
            }
            0x03 => {
                let len = client.read_u8().await? as usize;
                let mut domain = vec![0u8; len];
                client.read_exact(&mut domain).await?;
                String::from_utf8(domain).context("non-UTF-8 SOCKS5 target domain")?
            }
            0x04 => {
                let mut ip = [0u8; 16];
                client.read_exact(&mut ip).await?;
                IpAddr::V6(Ipv6Addr::from(ip)).to_string()
            }
            other => {
                let _ = send_socks5_reply(client, 0x08).await;
                bail!("unsupported SOCKS5 address type {other}");
            }
        };

        let mut port_buf = [0u8; 2];
        client.read_exact(&mut port_buf).await?;
        let target_port = u16::from_be_bytes(port_buf);

        let exit_proxy = {
            let guard = self
                .endpoints
                .read()
                .expect("ChainBridge endpoints read lock");
            guard.get(&proxy_id).cloned()
        };

        let Some(exit_proxy) = exit_proxy else {
            let _ = send_socks5_reply(client, 0x04).await;
            bail!("unknown proxy id {}", proxy_id);
        };

        Ok((exit_proxy, target_host, target_port))
    }
}

impl Drop for ChainBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn send_socks5_reply(stream: &mut TcpStream, rep: u8) -> io::Result<()> {
    let reply = [0x05, rep, 0x00, 0x01, 0, 0, 0, 0, 0, 0];
    stream.write_all(&reply).await?;
    stream.flush().await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::endpoint::ParsedProxy;
    use axum::Router;
    use axum::routing::get;
    use tokio::io::{AsyncBufReadExt, BufReader};

    #[test]
    fn base64_encodes_rfc4648_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"user:pass"), "dXNlcjpwYXNz");
    }

    async fn spawn_target_http_server() -> SocketAddr {
        let app = Router::new().route("/ping", get(|| async { "pong-from-target" }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        addr
    }

    async fn spawn_mock_socks5_server(expected_auth: Option<(&'static str, &'static str)>) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut client, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut hdr = [0u8; 2];
                    if client.read_exact(&mut hdr).await.is_err() {
                        return;
                    }
                    let mut methods = vec![0u8; hdr[1] as usize];
                    let _ = client.read_exact(&mut methods).await;
                    if let Some((exp_u, exp_p)) = expected_auth {
                        let _ = client.write_all(&[0x05, 0x02]).await;
                        let mut ah = [0u8; 2];
                        let _ = client.read_exact(&mut ah).await;
                        let mut u = vec![0u8; ah[1] as usize];
                        let _ = client.read_exact(&mut u).await;
                        let plen = client.read_u8().await.unwrap() as usize;
                        let mut p = vec![0u8; plen];
                        let _ = client.read_exact(&mut p).await;
                        if u != exp_u.as_bytes() || p != exp_p.as_bytes() {
                            let _ = client.write_all(&[0x01, 0x01]).await;
                            return;
                        }
                        let _ = client.write_all(&[0x01, 0x00]).await;
                    } else {
                        let _ = client.write_all(&[0x05, 0x00]).await;
                    }
                    let mut req = [0u8; 4];
                    if client.read_exact(&mut req).await.is_err() {
                        return;
                    }
                    let host = match req[3] {
                        0x01 => {
                            let mut ip = [0u8; 4];
                            let _ = client.read_exact(&mut ip).await;
                            IpAddr::V4(Ipv4Addr::from(ip)).to_string()
                        }
                        0x03 => {
                            let len = client.read_u8().await.unwrap() as usize;
                            let mut d = vec![0u8; len];
                            let _ = client.read_exact(&mut d).await;
                            String::from_utf8(d).unwrap()
                        }
                        _ => return,
                    };
                    let mut pb = [0u8; 2];
                    let _ = client.read_exact(&mut pb).await;
                    let port = u16::from_be_bytes(pb);
                    if let Ok(mut remote) = TcpStream::connect((host.as_str(), port)).await {
                        let _ = client
                            .write_all(&[0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
                            .await;
                        let _ = tokio::io::copy_bidirectional(&mut client, &mut remote).await;
                    } else {
                        let _ = client
                            .write_all(&[0x05, 0x05, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
                            .await;
                    }
                });
            }
        });
        addr
    }

    async fn spawn_mock_socks4_server() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut client, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut hdr = [0u8; 8];
                    if client.read_exact(&mut hdr).await.is_err() || hdr[0] != 0x04 {
                        return;
                    }
                    let port = u16::from_be_bytes([hdr[2], hdr[3]]);
                    // Read null-terminated user id
                    while client.read_u8().await.unwrap_or(0) != 0 {}
                    let host = if hdr[4] == 0 && hdr[5] == 0 && hdr[6] == 0 && hdr[7] != 0 {
                        let mut dom = Vec::new();
                        while let Ok(b) = client.read_u8().await {
                            if b == 0 {
                                break;
                            }
                            dom.push(b);
                        }
                        String::from_utf8(dom).unwrap()
                    } else {
                        Ipv4Addr::new(hdr[4], hdr[5], hdr[6], hdr[7]).to_string()
                    };
                    if let Ok(mut remote) = TcpStream::connect((host.as_str(), port)).await {
                        let _ = client.write_all(&[0x00, 0x5A, 0, 0, 0, 0, 0, 0]).await;
                        let _ = tokio::io::copy_bidirectional(&mut client, &mut remote).await;
                    } else {
                        let _ = client.write_all(&[0x00, 0x5B, 0, 0, 0, 0, 0, 0]).await;
                    }
                });
            }
        });
        addr
    }

    async fn spawn_mock_http_connect_server() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut reader = BufReader::new(client);
                    let mut req_line = String::new();
                    if reader.read_line(&mut req_line).await.is_err() {
                        return;
                    }
                    let target = req_line.split_whitespace().nth(1).unwrap_or("").to_string();
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).await.is_err() || line == "\r\n" || line.is_empty() {
                            break;
                        }
                    }
                    let mut client = reader.into_inner();
                    if let Ok(mut remote) = TcpStream::connect(&target).await {
                        let _ = client
                            .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                            .await;
                        let _ = tokio::io::copy_bidirectional(&mut client, &mut remote).await;
                    } else {
                        let _ = client
                            .write_all(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                            .await;
                    }
                });
            }
        });
        addr
    }

    #[tokio::test]
    async fn chain_bridge_tunnels_two_hops_via_wreq_across_all_protocols() {
        let target = spawn_target_http_server().await;
        let hop1_socks5 = spawn_mock_socks5_server(Some(("h1user", "h1pass"))).await;
        let hop2_socks5 = spawn_mock_socks5_server(None).await;
        let hop2_socks4 = spawn_mock_socks4_server().await;
        let hop2_http = spawn_mock_http_connect_server().await;

        let base_ep = ParsedProxy::parse_explicit(
            &format!("socks5h://h1user:h1pass@{hop1_socks5}"),
            true,
        )
        .unwrap()
        .into_endpoint(ProxyId(0));

        let bridge = ChainBridge::start(base_ep, Duration::from_secs(5))
            .await
            .unwrap();

        let eps = vec![
            ParsedProxy::parse_explicit(&format!("socks5h://{hop2_socks5}"), false)
                .unwrap()
                .into_endpoint(ProxyId(1)),
            ParsedProxy::parse_explicit(&format!("socks4a://{hop2_socks4}"), false)
                .unwrap()
                .into_endpoint(ProxyId(2)),
            ParsedProxy::parse_explicit(&format!("http://{hop2_http}"), false)
                .unwrap()
                .into_endpoint(ProxyId(3)),
        ];
        bridge.replace_endpoints(eps);

        let client = wreq::Client::builder().build().unwrap();
        let target_url = format!("http://{target}/ping");

        for id in [ProxyId(1), ProxyId(2), ProxyId(3)] {
            let proxy = bridge.wreq_proxy_for(id).unwrap();
            let resp = client.get(&target_url).proxy(proxy).send().await.unwrap();
            assert_eq!(resp.status(), 200);
            assert_eq!(resp.text().await.unwrap(), "pong-from-target");
            assert!(!bridge.is_hop1_failing());
        }
    }

    #[tokio::test]
    async fn chain_bridge_supports_http_hop1_rejects_bad_auth_and_isolates_hop1_vs_hop2_errors() {
        let target = spawn_target_http_server().await;
        let hop1_http = spawn_mock_http_connect_server().await;
        let hop2_socks5 = spawn_mock_socks5_server(None).await;

        let base_ep = ParsedProxy::parse_explicit(&format!("http://{hop1_http}"), true)
            .unwrap()
            .into_endpoint(ProxyId(0));
        let bridge = ChainBridge::start(base_ep, Duration::from_secs(2))
            .await
            .unwrap();

        let dead_port = {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            l.local_addr().unwrap().port()
        };
        bridge.register_endpoint(
            ParsedProxy::parse_explicit(&format!("socks5h://{hop2_socks5}"), false)
                .unwrap()
                .into_endpoint(ProxyId(10)),
        );
        bridge.register_endpoint(
            ParsedProxy::parse_explicit(&format!("socks5h://127.0.0.1:{dead_port}"), false)
                .unwrap()
                .into_endpoint(ProxyId(11)),
        );

        let client = wreq::Client::builder().build().unwrap();
        let resp = client
            .get(format!("http://{target}/ping"))
            .proxy(bridge.wreq_proxy_for(ProxyId(10)).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.text().await.unwrap(), "pong-from-target");

        // Unauthenticated / wrong-secret client must be rejected
        let bad_proxy = Proxy::all(format!(
            "socks5h://10:wrong-secret@127.0.0.1:{}",
            bridge.local_addr().port()
        ))
        .unwrap();
        assert!(
            client
                .get(format!("http://{target}/ping"))
                .proxy(bad_proxy)
                .send()
                .await
                .is_err()
        );

        // Hop 2 dead proxy fails without marking Hop 1 failing
        assert!(
            client
                .get(format!("http://{target}/ping"))
                .proxy(bridge.wreq_proxy_for(ProxyId(11)).unwrap())
                .send()
                .await
                .is_err()
        );
        assert!(!bridge.is_hop1_failing());

        // Now test unreachable Hop 1 marks hop1_failing == true
        let dead_hop1_port = {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            l.local_addr().unwrap().port()
        };
        let dead_base =
            ParsedProxy::parse_explicit(&format!("socks5://127.0.0.1:{dead_hop1_port}"), true)
                .unwrap()
                .into_endpoint(ProxyId(0));
        let bad_bridge = ChainBridge::start(dead_base, Duration::from_secs(2))
            .await
            .unwrap();
        bad_bridge.register_endpoint(
            ParsedProxy::parse_explicit(&format!("socks5h://{hop2_socks5}"), false)
                .unwrap()
                .into_endpoint(ProxyId(1)),
        );
        assert!(
            client
                .get(format!("http://{target}/ping"))
                .proxy(bad_bridge.wreq_proxy_for(ProxyId(1)).unwrap())
                .send()
                .await
                .is_err()
        );
        assert!(bad_bridge.is_hop1_failing());
        assert!(!bad_bridge.check_hop1_reachable().await);

        // Once a reachable Hop 1 listener appears on that port (or valid bridge checks Hop 1),
        // check_hop1_reachable clears hop1_failing automatically.
        let recoverable_bridge = ChainBridge::start(
            ParsedProxy::parse_explicit(&format!("http://{hop1_http}"), true)
                .unwrap()
                .into_endpoint(ProxyId(0)),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        recoverable_bridge.mark_hop1_error("simulated outage".to_string());
        assert!(recoverable_bridge.is_hop1_failing());
        assert!(recoverable_bridge.check_hop1_reachable().await);
        assert!(!recoverable_bridge.is_hop1_failing());
    }
}
