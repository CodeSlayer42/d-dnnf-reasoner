use std::cmp::{min, Ordering};
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions, read_to_string, remove_file};
use std::io::{BufRead, BufReader, Write};
use std::ops::Neg;
use std::path::Path;
use std::time::Instant;
use itertools::Itertools;
use nom::ParseTo;
use rug::ops::Pow;
use crate::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
use crate::ddnnf::anomalies::t_wise_sampling::t_iterator::TInteractionIter;
use streaming_iterator::StreamingIterator;
use crate::Ddnnf;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use crate::parser::build_ddnnf;


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SampleCorrectness { Ok, IncompleteConfig, UnorderedConfig, InvalidConfig, MissingInteraction }

impl Display for SampleCorrectness {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            SampleCorrectness::Ok => { "OK" }
            SampleCorrectness::IncompleteConfig => { "INCOMPLETE_CONFIG" }
            SampleCorrectness::UnorderedConfig => { "UNORDERED_CONFIG" }
            SampleCorrectness::InvalidConfig => { "INVALID_CONFIG" }
            SampleCorrectness::MissingInteraction => { "MISSING_INTERACTION" }
        };
        write!(f, "{str}")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SampleFitnessResult {
    fitness_sum: f64,
    fitness_avg: f64,
    fitness_std: f64
}

#[derive(Clone, Debug)]
pub struct SampleAdditionalInteractionsResult {
    interactions_counter: HashMap<Vec<i32>, usize>,
    n_additional_interactions_sum: usize,
    n_additional_interactions_avg: f64
}

#[derive(Copy, Clone, Debug)]
pub struct SampleAdditionalInteractionsFitnessResult {
    additional_interactions_fitness_sum: f64,
    additional_interactions_fitness_avg: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct SampleSelectedFeaturesResult {
    n_selected_features_sum: usize,
    n_selected_features_avg: f64,
    n_selected_features_std: f64
}


pub fn get_valid_interactions(ddnnf: &Ddnnf, t: usize) -> Vec<Vec<i32>> {
    let all_literals = (1..=ddnnf.number_of_variables as i32)
        .flat_map(|var| [var.neg(), var].into_iter())
        .collect_vec();

    let mut valid_interactions = Vec::new();
    TInteractionIter::new(&all_literals[..], t)
        .filter(|interaction| ddnnf.sat_immutable(interaction))
        .for_each(|interaction| valid_interactions.push(interaction.to_vec()));

    valid_interactions
}


pub fn check_correctness(sample: &Sample, ddnnf: &Ddnnf, valid_interactions: &Vec<Vec<i32>>) -> SampleCorrectness {
    for literals in sample.iter().map(|config| config.get_decided_literals().collect_vec()) {
        if ddnnf.number_of_variables as usize != literals.len() {
            return SampleCorrectness::IncompleteConfig;
        }

        if literals.iter().enumerate().any(|(idx, literal)| literal.unsigned_abs() != (idx + 1) as u32) {
            return SampleCorrectness::UnorderedConfig
        }

        if !ddnnf.sat_immutable(&literals[..]) {
            return SampleCorrectness::InvalidConfig;
        }
    }

    if valid_interactions.iter().any(|interaction| !sample.covers(interaction)) {
        return SampleCorrectness::MissingInteraction
    }

    SampleCorrectness::Ok
}


pub fn get_sample_fitness_result(sample: &Sample, ext_ddnnf: &ExtendedDdnnf) -> SampleFitnessResult {
    let sample_size = sample.len();

    let fitness_sum = sample.iter()
        .map(|config| ext_ddnnf.get_objective_fn_val_of_config(config))
        .sum();
    let fitness_avg: f64 = fitness_sum / sample_size as f64;
    let fitness_std_squared = sample.iter()
        .map(|config| ext_ddnnf.get_objective_fn_val_of_config(config))
        .map(|config_val| (config_val - fitness_avg).pow(2))
        .sum::<f64>() / sample_size as f64;

    SampleFitnessResult { fitness_sum, fitness_avg, fitness_std: fitness_std_squared.sqrt() }
}


pub fn get_sample_additional_interactions_result(sample: &Sample, valid_interactions: &Vec<Vec<i32>>, t: usize) -> SampleAdditionalInteractionsResult {
    let mut interactions_counter: HashMap<Vec<i32>, usize> = valid_interactions.iter().map(|interaction| (interaction.clone(), 0)).collect();

    for config in sample.iter() {
        TInteractionIter::new(config.get_literals(), min(config.get_n_decided_literals(), t))
            .for_each(|interaction| {
                interactions_counter.get_mut(interaction).map(|counter| *counter += 1);
            });
    }

    let n_additional_interactions_sum = interactions_counter.values().sum::<usize>() - valid_interactions.len();
    let n_additional_interactions_avg = n_additional_interactions_sum as f64 / sample.len() as f64;

    SampleAdditionalInteractionsResult { interactions_counter, n_additional_interactions_sum, n_additional_interactions_avg }
}


pub fn get_sample_additional_interactions_fitness_result(interactions_counter: &HashMap<Vec<i32>, usize>, ext_ddnnf: &ExtendedDdnnf) -> SampleAdditionalInteractionsFitnessResult {
    let additional_interactions_fitness_sum = interactions_counter.iter()
        .map(
            |(interaction, count)|
                ext_ddnnf.get_objective_fn_val_of_literals(&interaction[..]) * (count - 1) as f64
        )
        .sum();
    let additional_interactions_fitness_avg = additional_interactions_fitness_sum / interactions_counter.len() as f64;

    SampleAdditionalInteractionsFitnessResult { additional_interactions_fitness_sum, additional_interactions_fitness_avg }
}


pub fn get_sample_selected_features_result(sample: &Sample) -> SampleSelectedFeaturesResult {
    let n_selected_features_sum = sample.iter()
        .map(|config|
            config.get_decided_literals()
                .filter(|&literal| literal > 0)
                .count())
        .sum();
    let n_selected_features_avg: f64 = n_selected_features_sum as f64 / sample.len() as f64;
    let n_selected_features_std_squared = sample.iter()
        .map(|config|
            config.get_decided_literals()
                .filter(|&literal| literal > 0)
                .count())
        .map(|config_n_selected_features| (config_n_selected_features as f64 - n_selected_features_avg).pow(2))
        .sum::<f64>() / sample.len() as f64;

    SampleSelectedFeaturesResult { n_selected_features_sum, n_selected_features_avg, n_selected_features_std: n_selected_features_std_squared.sqrt() }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FM {
    dir: String,
    file: String,
    n_features: usize,
    n_clauses: usize,
}

impl PartialOrd<Self> for FM {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FM {
    fn cmp(&self, other: &Self) -> Ordering {
        self.n_features.cmp(&other.n_features)
    }
}

const EVAL_RESULTS_DIR: &str = "eval_results";
const FITNESS_DIR: &str = "fitness";
const FMS_DIR: &str = "fms";
const RESULTS_DIR: &str = "results";
const ANALYSIS_FILE: &str = "analysis.csv";
const ANALYSIS_LOG_FILE: &str = "analysis_log.txt";

const DDNNF_DIR: &str = "ddnnf";
const CNF_DIR: &str = "cnf";
const ORIGINAL_DIR: &str = "original";
const ADAPTED_DIR: &str = "adapted";
const YASA_DIR: &str = "yasa";
const OVERVIEW_FILE: &str = "overview.csv";
const SAMPLE_FILE: &str = "sample.txt";
const SAMPLING_TIME_FILE: &str = "time.txt";

const T: usize = 2;
const FITNESS_RUNS: usize = 5;
const YASA_RUNS: usize = 5;


pub fn analyse_eval_results() {
    let analysis_path = Path::new(EVAL_RESULTS_DIR).join(ANALYSIS_FILE);
    if analysis_path.exists() {
        remove_file(analysis_path.clone()).unwrap();
    }

    let mut analysis_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(analysis_path)
        .unwrap();


    let analysis_log_path = Path::new(EVAL_RESULTS_DIR).join(ANALYSIS_LOG_FILE);
    if analysis_log_path.exists() {
        remove_file(analysis_log_path.clone()).unwrap();
    }

    let mut analysis_log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(analysis_log_path)
        .unwrap();

    let fms = extract_fms(&mut analysis_log_file);
    writeln!(analysis_file, "{}", get_header()).unwrap();

    let n_fms = fms.len();
    for (fm_idx, (fm, ddnnf)) in fms.into_iter().enumerate() {
        writeln!(analysis_log_file, "Starting ({}/{}) FM {}/{} with {} features, {} clauses and {} nodes:", fm_idx+1, n_fms, fm.dir, fm.file, fm.n_features, fm.n_clauses, ddnnf.nodes.len()).unwrap();

        let mut ext_ddnnf = ExtendedDdnnf { ddnnf, attrs: HashMap::default(), objective_fn_vals: None };

        let now = Instant::now();
        let valid_interactions = get_valid_interactions(&ext_ddnnf.ddnnf, T);
        writeln!(analysis_log_file, "\tCalculated valid interactions in {} ms.", now.elapsed().as_millis()).unwrap();

        let fitness_vec = get_fitness(&fm.dir, &fm.file);

        let fm_stats = format!("{}; {}/{}; {}; {}; {}", fm_idx+1, fm.dir, fm.file, fm.n_features, fm.n_clauses, ext_ddnnf.ddnnf.nodes.len());
        writeln!(analysis_log_file, "\tStarting analysis of original algorithm results:").unwrap();
        let original_stats = analyse_original_algorithm_results(&fm.dir, &fm.file, &mut ext_ddnnf, &fitness_vec, &valid_interactions, &mut analysis_log_file);
        writeln!(analysis_log_file, "\tStarting analysis of adapted algorithm results:").unwrap();
        let adapted_stats = analyse_adapted_algorithm_results(&fm.dir, &fm.file, &mut ext_ddnnf, &fitness_vec, &valid_interactions, &mut analysis_log_file);
        writeln!(analysis_log_file, "\tStarting analysis of yasa algorithm results:").unwrap();
        let yasa_stats = analyse_yasa_algorithm_results(&fm.dir, &fm.file, &mut ext_ddnnf, &fitness_vec, &valid_interactions, &mut analysis_log_file);

        let data_row: String = vec![fm_stats, original_stats, adapted_stats, yasa_stats].into_iter()
            .intersperse("; ".to_string())
            .collect();
        writeln!(analysis_file, "{}", data_row).unwrap();
    }
}


pub fn extract_fms(analysis_log_file: &mut File) -> Vec<(FM, Ddnnf)> {
    let overview_path = Path::new(EVAL_RESULTS_DIR).join(FMS_DIR).join(OVERVIEW_FILE).into_os_string();
    let mut overview_file = File::open(overview_path).unwrap();

        BufReader::new(overview_file)
            .lines()
            .skip(1)
            .flatten()
            .map(|line| {
                let mut line_split = line.split(",");
                let mut fm_name_split = line_split.next().unwrap().split("/");

                let dir = fm_name_split.next().unwrap().to_string();
                let file = fm_name_split.next().unwrap().to_string();
                let n_features = line_split.next().unwrap().parse::<usize>().unwrap();
                let n_clauses = line_split.next().unwrap().parse::<usize>().unwrap();

                FM { dir, file, n_features, n_clauses }
            })
            .sorted()
            .enumerate()
            .map(|(fm_idx, fm)| {
                let ddnnf_path = Path::new(EVAL_RESULTS_DIR).join(FMS_DIR).join(DDNNF_DIR).join(&fm.dir).join(fm.file.clone() + ".nnf").into_os_string();
                let ddnnf: Ddnnf = build_ddnnf(OsStr::to_str(&ddnnf_path).unwrap(), Some(fm.n_features as u32));

                writeln!(analysis_log_file, "{};{}/{};{};{};{};", fm_idx+1, fm.dir, fm.file, fm.n_features, fm.n_clauses, ddnnf.nodes.len()).unwrap();

                (fm, ddnnf)
            })
            .collect_vec()
}


pub fn get_header() -> String {
    vec![
        get_fm_header(),
        get_algorithm_header("_original"),
        get_algorithm_header("_adapted"),
        get_algorithm_header("_yasa")
    ]
        .into_iter()
        .intersperse("; ".to_string())
        .collect()
}


pub fn get_fm_header() -> String {
    let header = vec!["fm_idx", "fm_name", "fm_n_features", "fm_n_clauses", "fm_n_ddnnf_nodes"];
    header.into_iter().intersperse("; ").collect()
}


pub fn get_algorithm_header(algorithm: &str) -> String {
    let header = vec![
        "correctness", "sampling_time", "sample_size", // ==> sample
        "fitness_sum", "fitness_avg", "fitness_std",    // ==> fitness - sample
        //"n_additional_interactions_sum", "n_additional_interactions_avg",   // ==> sample
        //"additional_interactions_fitness_sum", "additional_interactions_fitness_avg", "additional_interactions_fitness_std",    // ==> fitness - sample
        "n_selected_features_sum", "n_selected_features_avg", "n_selected_features_std" // ==> sample
    ];

    header.into_iter()
        .map(|column| column.to_string() + algorithm)
        .intersperse("; ".to_string())
        .collect()
}


pub fn analyse_original_algorithm_results(fm_dir: &str, fm_file: &str, ext_ddnnf: &mut ExtendedDdnnf, fitness_vec: &Vec<Vec<f64>>, valid_interactions: &Vec<Vec<i32>>, analysis_log_file: &mut File) -> String {
    let sampling_result_dir = Path::new(EVAL_RESULTS_DIR).join(RESULTS_DIR).join(ORIGINAL_DIR).join(fm_dir).join(fm_file);

    let mut correctness = Vec::new();
    let mut sample_size = Vec::new();
    let mut sample_time = Vec::new();

    let mut fitness_sum = Vec::new();
    let mut fitness_avg = Vec::new();
    let mut fitness_std = Vec::new();

    let mut n_selected_features_sum = Vec::new();
    let mut n_selected_features_avg = Vec::new();
    let mut n_selected_features_std = Vec::new();

    for _ in 0..=0 {
        writeln!(analysis_log_file, "\t\tStarting analysis of only sample:").unwrap();

        let sample_file = sampling_result_dir.join(SAMPLE_FILE);
        let sampling_time_file = sampling_result_dir.join(SAMPLING_TIME_FILE);

        if !sample_file.exists() {
            writeln!(analysis_log_file, "\t\t\tTimout for sample.").unwrap();

            correctness.push("Timeout".to_string());
            sample_size.push("Timeout".to_string());
            sample_time.push("Timeout".to_string());

            fitness_sum.push("Timeout".to_string());
            fitness_avg.push("Timeout".to_string());
            fitness_std.push("Timeout".to_string());

            n_selected_features_sum.push("Timeout".to_string());
            n_selected_features_avg.push("Timeout".to_string());
            n_selected_features_std.push("Timeout".to_string());

            continue;
        }

        let sample = parse_sample(sample_file.into_os_string(), false);

        let now = Instant::now();
        let is_correct = check_correctness(&sample, &ext_ddnnf.ddnnf, valid_interactions);
        correctness.push(is_correct.to_string());
        writeln!(analysis_log_file, "\t\t\tChecked correctness in {} ms.", now.elapsed().as_millis()).unwrap();
        if is_correct != SampleCorrectness::Ok {
            writeln!(analysis_log_file, "\t\t\tSample has error: {}", is_correct).unwrap();

            sample_size.push("ERR".to_string());
            sample_time.push("ERR".to_string());

            fitness_sum.push("ERR".to_string());
            fitness_avg.push("ERR".to_string());
            fitness_std.push("ERR".to_string());

            n_selected_features_sum.push("ERR".to_string());
            n_selected_features_avg.push("ERR".to_string());
            n_selected_features_std.push("ERR".to_string());

            continue;
        }

        sample_size.push(sample.len().to_string());
        sample_time.push(read_to_string(sampling_time_file).unwrap());

        let now = Instant::now();
        let selected_features_result = get_sample_selected_features_result(&sample);
        writeln!(analysis_log_file, "\t\t\tAnalysed selected features count in {} ms.", now.elapsed().as_millis()).unwrap();
        n_selected_features_sum.push(selected_features_result.n_selected_features_sum.to_string());
        n_selected_features_avg.push(selected_features_result.n_selected_features_avg.to_string());
        n_selected_features_std.push(selected_features_result.n_selected_features_std.to_string());


        for fitness_idx in 0..FITNESS_RUNS {
            let fitness_vals = fitness_vec[fitness_idx].clone();
            writeln!(analysis_log_file, "\t\t\tStarting analysis of fitness stats {}/{}:", fitness_idx+1, FITNESS_RUNS).unwrap();
            ext_ddnnf.objective_fn_vals = Some(fitness_vals);

            let now = Instant::now();
            let fitness_result = get_sample_fitness_result(&sample, &ext_ddnnf);
            writeln!(analysis_log_file, "\t\t\t\tAnalysed sample fitness in {} ms.", now.elapsed().as_millis()).unwrap();
            fitness_sum.push(fitness_result.fitness_sum.to_string());
            fitness_avg.push(fitness_result.fitness_avg.to_string());
            fitness_std.push(fitness_result.fitness_std.to_string());
        }
    }

    format!(
        "{:?}; {:?}; {:?}; \
        {:?}; {:?}; {:?}; \
        {:?}; {:?}; {:?}",
        correctness, sample_time, sample_size,
        fitness_sum, fitness_avg, fitness_std,
        n_selected_features_sum, n_selected_features_avg, n_selected_features_std
    )
}


pub fn analyse_adapted_algorithm_results(fm_dir: &str, fm_file: &str, ext_ddnnf: &mut ExtendedDdnnf, fitness_vec: &Vec<Vec<f64>>, valid_interactions: &Vec<Vec<i32>>, analysis_log_file: &mut File) -> String {
    let sampling_result_dir = Path::new(EVAL_RESULTS_DIR).join(RESULTS_DIR).join(ADAPTED_DIR).join(fm_dir).join(fm_file);

    let mut correctness = Vec::new();
    let mut sample_size = Vec::new();
    let mut sample_time = Vec::new();

    let mut fitness_sum = Vec::new();
    let mut fitness_avg = Vec::new();
    let mut fitness_std = Vec::new();

    let mut n_selected_features_sum = Vec::new();
    let mut n_selected_features_avg = Vec::new();
    let mut n_selected_features_std = Vec::new();

    for fitness_idx in 0..FITNESS_RUNS {
        writeln!(analysis_log_file, "\t\tStarting analysis of sample {}/{}.", fitness_idx+1, FITNESS_RUNS).unwrap();
        let sampling_result_dir = sampling_result_dir.join(format!("fitness{}.txt", fitness_idx+1));

        let sample_file = sampling_result_dir.join(SAMPLE_FILE);
        let sampling_time_file = sampling_result_dir.join(SAMPLING_TIME_FILE);

        if !sample_file.exists() {
            writeln!(analysis_log_file, "\t\t\tTimout for sample.").unwrap();

            correctness.push("Timeout".to_string());
            sample_size.push("Timeout".to_string());
            sample_time.push("Timeout".to_string());

            fitness_sum.push("Timeout".to_string());
            fitness_avg.push("Timeout".to_string());
            fitness_std.push("Timeout".to_string());

            n_selected_features_sum.push("Timeout".to_string());
            n_selected_features_avg.push("Timeout".to_string());
            n_selected_features_std.push("Timeout".to_string());

            continue;
        }

        let sample = parse_sample(sample_file.into_os_string(), false);

        let now = Instant::now();
        let is_correct = check_correctness(&sample, &ext_ddnnf.ddnnf, valid_interactions);
        correctness.push(is_correct.to_string());
        writeln!(analysis_log_file, "\t\t\tChecked correctness in {} ms.", now.elapsed().as_millis()).unwrap();
        if is_correct != SampleCorrectness::Ok {
            writeln!(analysis_log_file, "\t\t\tSample has error: {}", is_correct).unwrap();

            sample_size.push("ERR".to_string());
            sample_time.push("ERR".to_string());

            fitness_sum.push("ERR".to_string());
            fitness_avg.push("ERR".to_string());
            fitness_std.push("ERR".to_string());

            n_selected_features_sum.push("ERR".to_string());
            n_selected_features_avg.push("ERR".to_string());
            n_selected_features_std.push("ERR".to_string());

            continue;
        }

        sample_size.push(sample.len().to_string());
        sample_time.push(read_to_string(sampling_time_file).unwrap());

        let now = Instant::now();
        let selected_features_result = get_sample_selected_features_result(&sample);
        writeln!(analysis_log_file, "\t\t\tAnalysed selected features count in {} ms.", now.elapsed().as_millis()).unwrap();
        n_selected_features_sum.push(selected_features_result.n_selected_features_sum.to_string());
        n_selected_features_avg.push(selected_features_result.n_selected_features_avg.to_string());
        n_selected_features_std.push(selected_features_result.n_selected_features_std.to_string());


        let fitness_vals = fitness_vec[fitness_idx].clone();
        writeln!(analysis_log_file, "\t\t\tStarting analysis of next fitness stats:").unwrap();
        ext_ddnnf.objective_fn_vals = Some(fitness_vals);

        let now = Instant::now();
        let fitness_result = get_sample_fitness_result(&sample, &ext_ddnnf);
        writeln!(analysis_log_file, "\t\t\t\tAnalysed sample fitness in {} ms.", now.elapsed().as_millis()).unwrap();
        fitness_sum.push(fitness_result.fitness_sum.to_string());
        fitness_avg.push(fitness_result.fitness_avg.to_string());
        fitness_std.push(fitness_result.fitness_std.to_string());
    }

    format!(
        "{:?}; {:?}; {:?}; \
        {:?}; {:?}; {:?}; \
        {:?}; {:?}; {:?}",
        correctness, sample_time, sample_size,
        fitness_sum, fitness_avg, fitness_std,
        n_selected_features_sum, n_selected_features_avg, n_selected_features_std
    )
}


pub fn analyse_yasa_algorithm_results(fm_dir: &str, fm_file: &str, ext_ddnnf: &mut ExtendedDdnnf, fitness_vec: &Vec<Vec<f64>>, valid_interactions: &Vec<Vec<i32>>, analysis_log_file: &mut File) -> String {
    let sampling_result_dir = Path::new(EVAL_RESULTS_DIR).join(RESULTS_DIR).join(YASA_DIR).join(fm_dir).join(fm_file);

    let mut correctness = Vec::new();
    let mut sample_size = Vec::new();
    let mut sample_time = Vec::new();

    let mut fitness_sum_mat = Vec::new();
    let mut fitness_avg_mat = Vec::new();
    let mut fitness_std_mat = Vec::new();

    let mut n_selected_features_sum = Vec::new();
    let mut n_selected_features_avg = Vec::new();
    let mut n_selected_features_std = Vec::new();

    for sample_idx in 1..=YASA_RUNS {

        let mut fitness_sum = Vec::new();
        let mut fitness_avg = Vec::new();
        let mut fitness_std = Vec::new();

        writeln!(analysis_log_file, "\t\tStarting analysis of sample {}/{}.", sample_idx, YASA_RUNS).unwrap();
        let sampling_result_dir = sampling_result_dir.join(sample_idx.to_string());

        let sample_file = sampling_result_dir.join(SAMPLE_FILE);
        let sampling_time_file = sampling_result_dir.join(SAMPLING_TIME_FILE);

        if !sample_file.exists() {
            writeln!(analysis_log_file, "\t\t\tTimout for sample.").unwrap();

            correctness.push("Timeout".to_string());
            sample_size.push("Timeout".to_string());
            sample_time.push("Timeout".to_string());

            fitness_sum.push("Timeout".to_string());
            fitness_avg.push("Timeout".to_string());
            fitness_std.push("Timeout".to_string());

            n_selected_features_sum.push("Timeout".to_string());
            n_selected_features_avg.push("Timeout".to_string());
            n_selected_features_std.push("Timeout".to_string());

            continue;
        }

        let sample = parse_sample(sample_file.into_os_string(), true);

        let now = Instant::now();
        let is_correct = check_correctness(&sample, &ext_ddnnf.ddnnf, valid_interactions);
        correctness.push(is_correct.to_string());
        writeln!(analysis_log_file, "\t\t\tChecked correctness in {} ms.", now.elapsed().as_millis()).unwrap();
        if is_correct != SampleCorrectness::Ok {
            writeln!(analysis_log_file, "\t\t\tSample has error: {}", is_correct).unwrap();

            sample_size.push("ERR".to_string());
            sample_time.push("ERR".to_string());

            fitness_sum.push("ERR".to_string());
            fitness_avg.push("ERR".to_string());
            fitness_std.push("ERR".to_string());

            n_selected_features_sum.push("ERR".to_string());
            n_selected_features_avg.push("ERR".to_string());
            n_selected_features_std.push("ERR".to_string());

            continue;
        }

        sample_size.push(sample.len().to_string());
        sample_time.push(read_to_string(sampling_time_file).unwrap());

        let now = Instant::now();
        let selected_features_result = get_sample_selected_features_result(&sample);
        writeln!(analysis_log_file, "\t\t\tAnalysed selected features count in {} ms.", now.elapsed().as_millis()).unwrap();
        n_selected_features_sum.push(selected_features_result.n_selected_features_sum.to_string());
        n_selected_features_avg.push(selected_features_result.n_selected_features_avg.to_string());
        n_selected_features_std.push(selected_features_result.n_selected_features_std.to_string());


        for fitness_idx in 0..FITNESS_RUNS {
            let fitness_vals = fitness_vec[fitness_idx].clone();
            writeln!(analysis_log_file, "\t\t\tStarting analysis of fitness stats {}/{}:", fitness_idx+1, FITNESS_RUNS).unwrap();
            ext_ddnnf.objective_fn_vals = Some(fitness_vals);

            let now = Instant::now();
            let fitness_result = get_sample_fitness_result(&sample, &ext_ddnnf);
            writeln!(analysis_log_file, "\t\t\t\tAnalysed sample fitness in {} ms.", now.elapsed().as_millis()).unwrap();
            fitness_sum.push(fitness_result.fitness_sum.to_string());
            fitness_avg.push(fitness_result.fitness_avg.to_string());
            fitness_std.push(fitness_result.fitness_std.to_string());
        }

        fitness_sum_mat.push(fitness_sum);
        fitness_avg_mat.push(fitness_avg);
        fitness_std_mat.push(fitness_std);
    }

    format!(
        "{:?}; {:?}; {:?}; \
        {:?}; {:?}; {:?}; \
        {:?}; {:?}; {:?}",
        correctness, sample_time, sample_size,
        fitness_sum_mat, fitness_avg_mat, fitness_std_mat,
        n_selected_features_sum, n_selected_features_avg, n_selected_features_std
    )
}


pub fn parse_sample(path: OsString, is_yasa: bool) -> Sample {
    let configs = BufReader::new(File::open(path).unwrap())
        .lines()
        .flatten()
        .map(|line| {
            let mut line_split = line.split(",");
            let sample_str = line_split.skip(1).next().unwrap();

            let literals = sample_str.split(" ")
                .skip(if is_yasa { 1 } else { 0 })
                .map(|literal| {
                    let mut literal = literal.parse::<i32>().unwrap();
                    if is_yasa {
                        if literal < 0 {
                            literal += 1;
                        } else {
                            literal -= 1;
                        }
                    }

                    literal
                })
                .collect_vec();

            Config::from(&*literals, literals.len())
        })
        .collect_vec();

    Sample::new_from_configs(configs)
}


pub fn get_fitness(fm_dir: &str, fm_file: &str) -> Vec<Vec<f64>> {
    let fitness_dir = Path::new(EVAL_RESULTS_DIR).join(FITNESS_DIR).join(fm_dir).join(fm_file);
    let mut fitness_vec = Vec::new();

    for fitness_idx in 1..=FITNESS_RUNS {
        let file_name = format!("fitness{}.txt", fitness_idx);
        let fitness_vals = read_to_string(fitness_dir.join(file_name)).unwrap()
            .split(" ")
            .map(|fitness| fitness.parse::<f64>().unwrap())
            .collect_vec();
        fitness_vec.push(fitness_vals);
    }

    fitness_vec
}


#[cfg(test)]
mod test {
    use crate::ddnnf::anomalies::t_wise_sampling::sampling_eval::analyse_eval_results;

    #[test]
    fn run_eval_analysis() {
        analyse_eval_results();
    }
}