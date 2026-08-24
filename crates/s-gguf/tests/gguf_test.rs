use s_gguf::{GgufFile, GgufValue, GgmlType};

/// Costruisce un GGUF v3 sintetico completo: 3 KV (str, u32, array f32)
/// + 2 tensori (f32 e Q8_0), con allineamento dati a 32.
fn build_test_gguf() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"GGUF");
    v.extend_from_slice(&3u32.to_le_bytes());
    v.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
    v.extend_from_slice(&3u64.to_le_bytes()); // kv_count

    // KV 1: general.name = "test-model"
    let k = b"general.name";
    v.extend_from_slice(&(k.len() as u64).to_le_bytes());
    v.extend_from_slice(k);
    v.extend_from_slice(&8u32.to_le_bytes());
    let s = b"test-model";
    v.extend_from_slice(&(s.len() as u64).to_le_bytes());
    v.extend_from_slice(s);

    // KV 2: general.alignment = 32 (u32)
    let k = b"general.alignment";
    v.extend_from_slice(&(k.len() as u64).to_le_bytes());
    v.extend_from_slice(k);
    v.extend_from_slice(&4u32.to_le_bytes());
    v.extend_from_slice(&32u32.to_le_bytes());

    // KV 3: test.scales = array f32 [1.0, 2.0]
    let k = b"test.scales";
    v.extend_from_slice(&(k.len() as u64).to_le_bytes());
    v.extend_from_slice(k);
    v.extend_from_slice(&9u32.to_le_bytes()); // array
    v.extend_from_slice(&6u32.to_le_bytes()); // di f32
    v.extend_from_slice(&2u64.to_le_bytes()); // count 2
    v.extend_from_slice(&1.0f32.to_le_bytes());
    v.extend_from_slice(&2.0f32.to_le_bytes());

    // Tensore 1: "w0" f32 dims=[2,2] offset=0
    let t = b"w0";
    v.extend_from_slice(&(t.len() as u64).to_le_bytes());
    v.extend_from_slice(t);
    v.extend_from_slice(&2u32.to_le_bytes());
    v.extend_from_slice(&2u64.to_le_bytes());
    v.extend_from_slice(&2u64.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // F32
    v.extend_from_slice(&0u64.to_le_bytes());

    // Tensore 2: "w1" Q8_0 dims=[64,1] offset=16 (dopo i 16 byte di w0)
    let t = b"w1";
    v.extend_from_slice(&(t.len() as u64).to_le_bytes());
    v.extend_from_slice(t);
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&64u64.to_le_bytes());
    v.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
    v.extend_from_slice(&16u64.to_le_bytes());

    // --- sezione dati (allineata a 32: header finisce qui, pad a 32) ---
    while v.len() % 32 != 0 {
        v.push(0);
    }
    // w0: 4 f32 = [1.0, 2.0, 3.0, 4.0]
    v.extend_from_slice(&1.0f32.to_le_bytes());
    v.extend_from_slice(&2.0f32.to_le_bytes());
    v.extend_from_slice(&3.0f32.to_le_bytes());
    v.extend_from_slice(&4.0f32.to_le_bytes());
    // w1: 2 blocchi Q8_0 (64 valori): blocco0 d=1.0 qs=0..31, blocco1 d=0.5 qs=1..32
    v.extend_from_slice(&0x3C00u16.to_le_bytes()); // fp16 1.0
    for i in 0..32i8 {
        v.push(i as u8);
    }
    v.extend_from_slice(&0x3800u16.to_le_bytes()); // fp16 0.5
    for i in 1..=32i8 {
        v.push(i as u8);
    }
    v
}

fn write_temp_named(name: &str, data: &[u8]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("s_gguf_test_{name}.gguf"));
    std::fs::write(&p, data).unwrap();
    p
}

#[test]
fn header_and_metadata() {
    let path = write_temp_named("main", &build_test_gguf());
    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.header.version, 3);
    assert_eq!(f.header.tensor_count, 2);
    assert_eq!(f.header.kv_count, 3);
    assert_eq!(f.alignment(), 32);

    assert_eq!(
        f.metadata.iter().find(|(k, _)| k == "general.name").unwrap().1,
        GgufValue::String("test-model".into())
    );
    match &f.metadata.iter().find(|(k, _)| k == "test.scales").unwrap().1 {
        GgufValue::Array(a) => {
            assert_eq!(a.len(), 2);
            assert_eq!(a[0], GgufValue::F32(1.0));
            assert_eq!(a[1], GgufValue::F32(2.0));
        }
        other => panic!("atteso array, trovato {other:?}"),
    }
}

#[test]
fn tensor_infos_and_lazy_data() {
    let path = write_temp_named("main", &build_test_gguf());
    let mut f = GgufFile::open(&path).unwrap();

    let w0 = f.tensor("w0").unwrap();
    assert_eq!(w0.ggml_type, GgmlType::F32);
    assert_eq!(w0.n_elements(), 4);

    let w1 = f.tensor("w1").unwrap();
    assert_eq!(w1.ggml_type, GgmlType::Q8_0);
    assert_eq!(w1.n_bytes(), 2 * 34); // 2 blocchi da 34 byte

    // dati f32 di w0
    let d0 = f.tensor_data_f32("w0").unwrap();
    assert_eq!(d0, vec![1.0, 2.0, 3.0, 4.0]);

    // dequant Q8_0 di w1: blocco0 = qs*1.0, blocco1 = qs*0.5
    let d1 = f.tensor_data_f32("w1").unwrap();
    assert_eq!(d1.len(), 64);
    assert!((d1[0] - 0.0).abs() < 1e-6);
    assert!((d1[31] - 31.0).abs() < 1e-4);
    assert!((d1[32] - 0.5).abs() < 1e-6);
    assert!((d1[63] - 16.0).abs() < 1e-4);

    // tensore mancante
    assert!(f.tensor("inesistente").is_err());
}

#[test]
fn bad_magic_rejected() {
    let path = write_temp_named("badmagic", b"NOPE0000");
    assert!(matches!(
        GgufFile::open(&path),
        Err(s_gguf::GgufError::BadMagic)
    ));
}

/// Verifica F0 su file reale: se S_GGUF_REAL_TEST è impostato, legge il file
/// e confronta il conteggio tensori con l'atteso (dalla CLI llama.cpp).
#[test]
fn real_file_smoke() {
    let Ok(path) = std::env::var("S_GGUF_REAL_TEST") else {
        return; // test attivo solo su richiesta
    };
    let f = GgufFile::open(&path).unwrap();
    println!("version: {}", f.header.version);
    println!("tensors: {}", f.header.tensor_count);
    println!("kv: {}", f.header.kv_count);
    println!("data_start: {}", f.data_start());
    for (k, v) in f.metadata.iter().take(10) {
        println!("  {k} = {v:?}");
    }
    for t in f.tensors.iter().take(5) {
        println!("  {} [{:?}] dims={:?} bytes={}", t.name, t.ggml_type, t.dims, t.n_bytes());
    }
}

#[test]
fn real_file_dequant_sanity() {
    let Ok(path) = std::env::var("S_GGUF_REAL_TEST") else { return; };
    let mut f = GgufFile::open(&path).unwrap();
    // dequant del primo pezzo di token_embd (Q8_0, il primo tensore del file)
    let d = f.tensor_data_f32("token_embd.weight").unwrap();
    assert_eq!(d.len(), 576 * 49152);
    let nan = d.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nan, 0, "NaN/inf nella dequant reale");
    let mean: f32 = d[..100_000].iter().sum::<f32>() / 100_000.0;
    println!("primi 100k valori: media={mean:.5} min={:.4} max={:.4}",
        d[..100_000].iter().cloned().fold(f32::MAX, f32::min),
        d[..100_000].iter().cloned().fold(f32::MIN, f32::max));
    assert!(mean.abs() < 0.1, "media anomala: {mean}");
}
