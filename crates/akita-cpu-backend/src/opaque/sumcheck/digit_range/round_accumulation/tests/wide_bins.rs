#[test]
fn p64_wide_bins_match_direct_exact_reduction() {
    check_blocked_accumulation_with_inner_len::<Ext2<Prime64Offset59>>(1024);
}
