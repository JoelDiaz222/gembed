use std::cell::RefCell;
use std::io::{BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

const LOG_PATH: &str = "/dev/shm/gembed_telemetry_log";

thread_local! {
    static WRITER: RefCell<Option<BufWriter<std::fs::File>>> = RefCell::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(LOG_PATH)
            .ok()
            .map(BufWriter::new),
    );
}

pub fn tlog(label: &str, n: usize) {
    let us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);

    WRITER.with(|cell| {
        if let Some(ref mut w) = *cell.borrow_mut() {
            let _ = writeln!(w, "{}\t{}\t{}", us, label, n);
            let _ = w.flush();
        }
    });
}
