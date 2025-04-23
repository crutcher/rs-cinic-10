pub mod images;
pub mod index;

pub use index::Cinic10Index;

use std::env;
use std::path::PathBuf;
use std::sync::RwLock;

static STATIC_DEFAULT_PATH: RwLock<Option<PathBuf>> = RwLock::new(None);
pub const CINC10_PATH_ENV_VAR: &str = "CINIC10_PATH";

/// Get the default path for CINIC-10 data.
///
/// Returns either the last call to `set_default_path()`;
/// or the value of the `CINIC10_PATH` env var,
/// or `None`.
pub fn get_default_data_path() -> Option<PathBuf> {
    let path = STATIC_DEFAULT_PATH.read().unwrap().clone();
    if path.is_some() {
        return path;
    }

    match env::var(CINC10_PATH_ENV_VAR) {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => None,
    }
}

pub fn default_data_path_or_panic() -> PathBuf {
    get_default_data_path().unwrap_or_else(|| {
        panic!(
            "CINIC-10 data path not set. \
            Set the {} environment variable or call set_default_data_path()",
            CINC10_PATH_ENV_VAR
        )
    })
}

/// Set the value of `get_default_path()` in future calls.
pub fn set_default_data_path(path: Option<PathBuf>) {
    *STATIC_DEFAULT_PATH.write().unwrap() = path;
}
