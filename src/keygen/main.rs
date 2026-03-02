use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use ed25519_dalek::{Signer, SigningKey};
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        usage();
    }

    match args[1].as_str() {
        "--generate-keypair" => generate_keypair(&args),
        "--email" => sign_license(&args),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  keygen --generate-keypair [--output-dir <dir>]");
    eprintln!(
        "  keygen --email <email> --expires <YYYY-MM-DD> --tier <tier> [--private-key <path>] [--output <path>]"
    );
    process::exit(1);
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn generate_keypair(args: &[String]) {
    let output_dir = get_arg(args, "--output-dir").unwrap_or_else(|| "keys".into());
    fs::create_dir_all(&output_dir).expect("failed to create output directory");

    let mut csprng = rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();

    let private_path = format!("{}/private.key", output_dir);
    let public_path = format!("{}/public.key", output_dir);

    fs::write(&private_path, signing_key.to_bytes()).expect("failed to write private key");
    fs::write(&public_path, verifying_key.to_bytes()).expect("failed to write public key");

    eprintln!("Keypair generated:");
    eprintln!("  Private key: {}", private_path);
    eprintln!("  Public key:  {}", public_path);
}

fn sign_license(args: &[String]) {
    let email = get_arg(args, "--email").unwrap_or_else(|| {
        eprintln!("Error: --email is required");
        process::exit(1);
    });
    let expires = get_arg(args, "--expires").unwrap_or_else(|| {
        eprintln!("Error: --expires is required");
        process::exit(1);
    });
    let tier = get_arg(args, "--tier").unwrap_or_else(|| {
        eprintln!("Error: --tier is required");
        process::exit(1);
    });

    let private_key_path =
        get_arg(args, "--private-key").unwrap_or_else(|| "keys/private.key".into());
    let private_bytes = fs::read(&private_key_path).unwrap_or_else(|e| {
        eprintln!("Error reading private key '{}': {}", private_key_path, e);
        process::exit(1);
    });
    let key_bytes: [u8; 32] = private_bytes.try_into().unwrap_or_else(|_| {
        eprintln!("Error: private key must be exactly 32 bytes");
        process::exit(1);
    });
    let signing_key = SigningKey::from_bytes(&key_bytes);

    let payload = format!("email={}\nexpires={}\ntier={}", email, expires, tier);
    let signature = signing_key.sign(payload.as_bytes());
    let sig_b64 = BASE64.encode(signature.to_bytes());
    let license = format!("{}\n---\n{}", payload, sig_b64);

    if let Some(output_path) = get_arg(args, "--output") {
        if let Some(parent) = std::path::Path::new(&output_path).parent() {
            fs::create_dir_all(parent).expect("failed to create output directory");
        }
        fs::write(&output_path, &license).unwrap_or_else(|e| {
            eprintln!("Error writing license to '{}': {}", output_path, e);
            process::exit(1);
        });
        eprintln!("License written to: {}", output_path);
    } else {
        println!("{}", license);
    }
}
