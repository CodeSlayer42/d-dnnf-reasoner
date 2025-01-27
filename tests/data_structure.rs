extern crate ddnnf_lib;

use ddnnf_lib::ddnnf::Ddnnf;
use ddnnf_lib::parser;

use file_diff::diff_files;
use std::fs;
use std::fs::File;

#[test]
fn card_of_features_test() {
    let c2d_out = "./tests/data/auto1_c2d_fs.csv";
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf("./tests/data/auto1_c2d.nnf", None);
    ddnnf
        .card_of_each_feature(c2d_out)
        .unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(c2d_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(c2d_out);

    let d4_out = "./tests/data/auto1_d4_fs.csv";
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf("./tests/data/auto1_d4.nnf", Some(2513));
    ddnnf
        .card_of_each_feature(d4_out)
        .unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(d4_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(d4_out);
}

#[test]
fn card_of_pc_test() {
    let c2d_out = "./tests/data/auto1_c2d_pc.txt";
    let d4_out = "./tests/data/auto1_d4_pc.txt";
    let sb_file_path = "./tests/data/auto1_sb_pc.txt";
    let config_file = "./tests/data/auto1.config";

    let mut ddnnf: Ddnnf =
        parser::build_ddnnf("tests/data/auto1_c2d.nnf", None);
    ddnnf.max_worker = 1;
    ddnnf
        .card_multi_queries(config_file, c2d_out)
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(c2d_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    fs::remove_file(c2d_out).unwrap();

    let mut ddnnf: Ddnnf =
        parser::build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
    ddnnf.max_worker = 1;
    ddnnf
        .card_multi_queries(config_file, d4_out)
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(d4_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    fs::remove_file(d4_out).unwrap();
}

#[test]
fn heuristics_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf("./tests/data/auto1_c2d.nnf", None);
    ddnnf.print_all_heuristics();
}
