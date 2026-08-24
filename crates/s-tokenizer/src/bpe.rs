//! BPE merge loop (parte della logica principale in lib.rs; qui i test dedicati).

#[cfg(test)]
mod tests {
    use super::super::pre::bytes_to_unicode;

    #[test]
    fn byte_mapping_covers_all_256() {
        let m = bytes_to_unicode();
        assert_eq!(m.len(), 256);
        // tutti caratteri unicode validi e distinti
        let mut seen = std::collections::HashSet::new();
        for &c in &m {
            assert!(seen.insert(c), "carattere duplicato {c:?}");
        }
        // byte stampabile 'A' (65) mappa a se stesso
        assert_eq!(m[65], 'A');
        assert_eq!(m[33], '!');
    }
}
