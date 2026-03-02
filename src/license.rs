use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use ed25519_dalek::{Signature, VerifyingKey};

use std::fmt;
use std::time::SystemTime;

const PUBLIC_KEY_BYTES: &[u8; 32] = include_bytes!("../keys/public.key");

#[derive(Debug)]
pub struct License {
    pub email: String,
    pub expires: String,
    pub tier: String,
}

#[derive(Debug)]
pub enum LicenseError {
    NotFound,
    InvalidFormat(String),
    InvalidSignature,
    Expired { expires: String },
    IoError(String),
}

impl fmt::Display for LicenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LicenseError::NotFound => write!(
                f,
                "No license found. Set DIOPTRIN_LICENSE env var or place license at ~/.dioptrin/license.key"
            ),
            LicenseError::InvalidFormat(msg) => write!(f, "Invalid license format: {}", msg),
            LicenseError::InvalidSignature => write!(f, "Invalid license signature"),
            LicenseError::Expired { expires } => {
                write!(f, "License expired on {}", expires)
            }
            LicenseError::IoError(msg) => write!(f, "IO error reading license: {}", msg),
        }
    }
}

impl std::error::Error for LicenseError {}

/// Top-level license verification using the compiled-in public key.
pub fn verify_license() -> Result<License, LicenseError> {
    let content = read_license_content()?;
    let key = VerifyingKey::from_bytes(PUBLIC_KEY_BYTES)
        .map_err(|_| LicenseError::InvalidFormat("corrupted public key".into()))?;
    verify_with_key(&content, &key)
}

/// Verify a license string against a given public key. Useful for testing.
pub fn verify_with_key(content: &str, public_key: &VerifyingKey) -> Result<License, LicenseError> {
    let (payload, signature) = parse_license(content)?;
    verify_signature(payload.as_bytes(), &signature, public_key)?;
    let license = parse_payload(&payload)?;
    check_expiration(&license.expires)?;
    Ok(license)
}

fn read_license_content() -> Result<String, LicenseError> {
    // 1. Try DIOPTRIN_LICENSE env var (content, not path)
    if let Ok(content) = std::env::var("DIOPTRIN_LICENSE") {
        if !content.is_empty() {
            return Ok(content);
        }
    }

    // 2. Fall back to ~/.dioptrin/license.key
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| LicenseError::NotFound)?;
    let path = std::path::PathBuf::from(home)
        .join(".dioptrin")
        .join("license.key");
    std::fs::read_to_string(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            LicenseError::NotFound
        } else {
            LicenseError::IoError(e.to_string())
        }
    })
}

fn parse_license(content: &str) -> Result<(String, Signature), LicenseError> {
    let parts: Vec<&str> = content.splitn(2, "\n---\n").collect();
    if parts.len() != 2 {
        return Err(LicenseError::InvalidFormat(
            "missing '---' separator".into(),
        ));
    }
    let payload = parts[0].to_string();
    let sig_b64 = parts[1].trim();
    let sig_bytes = BASE64
        .decode(sig_b64)
        .map_err(|e| LicenseError::InvalidFormat(format!("invalid base64 signature: {}", e)))?;
    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|_| LicenseError::InvalidFormat("invalid signature length".into()))?;
    Ok((payload, signature))
}

fn verify_signature(
    payload: &[u8],
    signature: &Signature,
    public_key: &VerifyingKey,
) -> Result<(), LicenseError> {
    use ed25519_dalek::Verifier;
    public_key
        .verify(payload, signature)
        .map_err(|_| LicenseError::InvalidSignature)
}

fn parse_payload(payload: &str) -> Result<License, LicenseError> {
    let mut email = None;
    let mut expires = None;
    let mut tier = None;

    for line in payload.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "email" => email = Some(value.trim().to_string()),
                "expires" => expires = Some(value.trim().to_string()),
                "tier" => tier = Some(value.trim().to_string()),
                _ => {}
            }
        }
    }

    Ok(License {
        email: email
            .ok_or_else(|| LicenseError::InvalidFormat("missing 'email' field".into()))?,
        expires: expires
            .ok_or_else(|| LicenseError::InvalidFormat("missing 'expires' field".into()))?,
        tier: tier.ok_or_else(|| LicenseError::InvalidFormat("missing 'tier' field".into()))?,
    })
}

fn check_expiration(expires: &str) -> Result<(), LicenseError> {
    let exp_parts: Vec<&str> = expires.split('-').collect();
    if exp_parts.len() != 3 {
        return Err(LicenseError::InvalidFormat(
            "expires must be YYYY-MM-DD".into(),
        ));
    }
    let exp_year: u32 = exp_parts[0]
        .parse()
        .map_err(|_| LicenseError::InvalidFormat("invalid year".into()))?;
    let exp_month: u32 = exp_parts[1]
        .parse()
        .map_err(|_| LicenseError::InvalidFormat("invalid month".into()))?;
    let exp_day: u32 = exp_parts[2]
        .parse()
        .map_err(|_| LicenseError::InvalidFormat("invalid day".into()))?;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| LicenseError::IoError(e.to_string()))?;
    let secs = now.as_secs();
    // Simple date extraction from unix timestamp
    let days = secs / 86400;
    let (now_year, now_month, now_day) = days_to_date(days);

    let exp_tuple = (exp_year, exp_month, exp_day);
    let now_tuple = (now_year, now_month, now_day);

    if exp_tuple < now_tuple {
        return Err(LicenseError::Expired {
            expires: expires.to_string(),
        });
    }
    Ok(())
}

fn days_to_date(days_since_epoch: u64) -> (u32, u32, u32) {
    // Civil date from days since 1970-01-01 (Euclidean affine algorithm)
    let z = days_since_epoch as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn test_keypair() -> (SigningKey, VerifyingKey) {
        // Deterministic test keypair (DO NOT use in production)
        let secret: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let signing = SigningKey::from_bytes(&secret);
        let verifying = signing.verifying_key();
        (signing, verifying)
    }

    fn sign_license(payload: &str, signing_key: &SigningKey) -> String {
        use ed25519_dalek::Signer;
        let signature = signing_key.sign(payload.as_bytes());
        let sig_b64 = BASE64.encode(signature.to_bytes());
        format!("{}\n---\n{}", payload, sig_b64)
    }

    #[test]
    fn test_valid_license() {
        let (signing, verifying) = test_keypair();
        let payload = "email=test@example.com\nexpires=2099-12-31\ntier=pro";
        let content = sign_license(payload, &signing);
        let license = verify_with_key(&content, &verifying).unwrap();
        assert_eq!(license.email, "test@example.com");
        assert_eq!(license.expires, "2099-12-31");
        assert_eq!(license.tier, "pro");
    }

    #[test]
    fn test_expired_license() {
        let (signing, verifying) = test_keypair();
        let payload = "email=test@example.com\nexpires=2020-01-01\ntier=pro";
        let content = sign_license(payload, &signing);
        let err = verify_with_key(&content, &verifying).unwrap_err();
        assert!(matches!(err, LicenseError::Expired { .. }));
    }

    #[test]
    fn test_invalid_signature() {
        let (_signing, verifying) = test_keypair();
        // Use a different keypair to sign
        let other_secret: [u8; 32] = [99; 32];
        let other_signing = SigningKey::from_bytes(&other_secret);
        let payload = "email=test@example.com\nexpires=2099-12-31\ntier=pro";
        let content = sign_license(payload, &other_signing);
        let err = verify_with_key(&content, &verifying).unwrap_err();
        assert!(matches!(err, LicenseError::InvalidSignature));
    }

    #[test]
    fn test_tampered_payload() {
        let (signing, _verifying) = test_keypair();
        let payload = "email=test@example.com\nexpires=2099-12-31\ntier=pro";
        let content = sign_license(payload, &signing);
        // Tamper with the payload
        let tampered = content.replace("pro", "enterprise");
        let (_, verifying) = test_keypair();
        let err = verify_with_key(&tampered, &verifying).unwrap_err();
        assert!(matches!(err, LicenseError::InvalidSignature));
    }

    #[test]
    fn test_missing_separator() {
        let (_, verifying) = test_keypair();
        let err = verify_with_key("no separator here", &verifying).unwrap_err();
        assert!(matches!(err, LicenseError::InvalidFormat(_)));
    }

    #[test]
    fn test_missing_fields() {
        let (signing, verifying) = test_keypair();
        let payload = "email=test@example.com\nexpires=2099-12-31";
        let content = sign_license(payload, &signing);
        let err = verify_with_key(&content, &verifying).unwrap_err();
        assert!(matches!(err, LicenseError::InvalidFormat(_)));
    }

    #[test]
    fn test_days_to_date() {
        assert_eq!(days_to_date(0), (1970, 1, 1));
        assert_eq!(days_to_date(365), (1971, 1, 1)); // 1970 is not a leap year
        assert_eq!(days_to_date(18627), (2020, 12, 31));
        assert_eq!(days_to_date(18628), (2021, 1, 1));
        assert_eq!(days_to_date(19357), (2022, 12, 31));
    }
}
