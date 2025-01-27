#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::{Parser, ArgGroup};

use colour::red;

use std::io::{self, Write, BufRead};
use std::path::Path;
use std::process::exit;
use std::sync::mpsc::{self, Receiver};
use std::thread::{self};
use std::time::Instant;

use ddnnf_lib::ddnnf::Ddnnf;
use ddnnf_lib::parser::{self as dparser, persisting::write_ddnnf};

#[derive(Parser)]
#[command(author, version, about, arg_required_else_help(true),
help_template("\
{before-help}{name} {version}
{author-with-newline}{about-with-newline}
{usage-heading} {usage}

{all-args}{after-help}
"), allow_negative_numbers = true, long_about = None)]

#[clap(group(
    ArgGroup::new("loading")
        .required(true)
        .args(&["file_path", "pipe_ddnnf_stdin"]),
))]

struct Cli {
    /// The path to the file in dimacs format. The d-dnnf has to be either fulfill the requirements
    /// of the c2d format and be smooth or produced by the newest d4 compiler version to work properly!
    #[arg(verbatim_doc_comment)]
    file_path: Option<String>,

    /// Allows to load the ddnnf via stdin.
    /// Either ommited_features has to be set or the file must start with a header of the form 'nnf n v e',
    /// where v is the number of nodes, e is the number of edges,
    /// and n is the number of variables over which the d-dnnf is defined.
    /// Like the c2d and the d4 format specifies, each line must be defided by a new line.
    /// Two following new lines end the reading from stdin.
    #[arg(short, long, verbatim_doc_comment)]    
    pipe_ddnnf_stdin: bool,

    /// The numbers of the features that should be included or excluded (positive number to include, negative to exclude).
    /// Can be one or multiple. A feature f has to be ∈ ℤ and the only allowed seperator is a whitespace!
    /// This should be the last option, because of the ambiguity of a hyphen as signed int and another option or flag.
    #[arg(short, long, num_args = 0.., verbatim_doc_comment)]
    features: Option<Vec<i32>>,

    /// The stream API
    #[arg(long)]
    stream: bool,

    /// Computes the cardinality of features for the feature model,
    /// i.e. the cardinality iff we select one feature for all features.
    /// Default output file is '{FILE_NAME}-features.csv'.
    #[clap(short, long, num_args = 0..=1, verbatim_doc_comment)]
    card_of_fs: Option<Vec<String>>,

    /// Mulitple queries that follow the feature format and the qeries themself are seperatated by \"\\n\".
    /// The option takes a file with queries as first argument and an optional second argument for saving the results.
    /// Default output file is '{FILE_NAME}-queries.txt'.
    #[arg(short, long, num_args = 1..=2, verbatim_doc_comment)]
    queries: Option<Vec<String>>,

    /// Computes core, dead, false-optional features, and atomic sets.
    /// You can add a file path for saving the information.
    /// Alternativly, the information is saved in anomalies.txt
    #[arg(short, long, num_args = 0..=1, verbatim_doc_comment)]
    anomalies: Option<Vec<String>>,   

    /// The number of omitted features.
    /// This is strictly necessary if the ddnnf has the d4 format respectivily does not contain a header.
    #[arg(short, long, verbatim_doc_comment)]
    ommited_features: Option<u32>,

    /// Save the smooth ddnnf in the c2d format. Default output file is '{FILE_NAME}.nnf'.
    /// Alternatively, you can choose a name. The .nnf ending is added automatically.
    #[arg(short, long, num_args = 0..=1, verbatim_doc_comment)]
    save_ddnnf: Option<Vec<String>>,

    /// Specify how many threads should be used. Default is 4.
    /// Possible values are between 1 and 32.
    #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 4, verbatim_doc_comment)]
    jobs: u16,

    /// Provides information about the type of nodes, their connection and the different paths.
    #[arg(long, verbatim_doc_comment)]
    heuristics: bool, 
}

fn main() {
    let cli = Cli::parse();

    // create the ddnnf based of the input file that is required
    let time = Instant::now();
    let mut ddnnf: Ddnnf;

    if cli.pipe_ddnnf_stdin {
        // read model line by line from stdin
        let mut input = Vec::new();
        for line in io::stdin().lock().lines() {
            let read_line = line.unwrap();
            if read_line.is_empty() { break; }
            input.push(read_line);
        }
        ddnnf = dparser::distribute_building(input, cli.ommited_features);
    } else {
        let ddnnf_path = &cli.file_path.clone().unwrap();
        ddnnf = dparser::build_ddnnf(ddnnf_path, cli.ommited_features)
    }

    // print additional output, iff we are not in the stream mode
    if !cli.stream {
        let elapsed_time = time.elapsed().as_secs_f32();
        println!(
            "Ddnnf overall count: {:#?}\nElapsed time for parsing and overall count in seconds: {:.3}s.",
            ddnnf.rc(),
            elapsed_time
        );
    }

    // print the heuristics
    if cli.heuristics {
        ddnnf.print_all_heuristics();
    }

    // change the number of threads used for cardinality of features and partial configurations
    ddnnf.max_worker = cli.jobs;

    // computes the cardinality for the partial configuration that can be mentioned with parameters
    if cli.features.is_some() {
        let features = cli.features.unwrap();
        println!("\nDdnnf count for query {:?} is: {:?}", &features, ddnnf.execute_query(&features));
        let marked_nodes = ddnnf.get_marked_nodes_clone(&features);
        println!("While computing the cardinality of the partial configuration {} out of the {} nodes were marked. \
            That are {:.2}%", marked_nodes.len(), ddnnf.nodes.len(), marked_nodes.len() as f64 / ddnnf.nodes.len() as f64 * 100.0);
    }

    // file path without last extension
    let file_path = String::from(
        Path::new(&cli.file_path.unwrap_or(String::from("ddnnf.nnf")))
        .with_extension("").file_name().unwrap().to_str().unwrap());

    // computes the cardinality of partial configurations and saves the results in a .txt file
    // the results do not have to be in the same order if the number of threads is greater than one
    if cli.queries.is_some() {
        let params: Vec<String> = cli.queries.unwrap();
        if !Path::new(&params[0]).exists() {
            red!("error:");
            eprintln!("{} is NOT a valid file path! Exiting...", params[0]);
            exit(1);
        }
        let config_path = &params[0];
        let file_path_out = &format!("{}-queries.txt",
            if params.len() == 2 {
                &params[1]
            } else {
                &file_path
            });

        let time = Instant::now();
        ddnnf
            .card_multi_queries(config_path, file_path_out)
            .unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();

        println!(
            "\nComputed values of all queries in {} and the results are saved in {}\n\
            It took {} seconds. That is an average of {} seconds per query",
            config_path,
            file_path_out,
            elapsed_time,
            elapsed_time / dparser::parse_queries_file(config_path.as_str()).len() as f64
        );
    }

    // computes the cardinality of features and saves the results in a .csv file
    // the cardinalities are always sorted from lowest to highest (also for multiple threads)
    if cli.card_of_fs.is_some() {
        let features_path = build_file_path(cli.card_of_fs, &file_path, "-features.csv");

        let time = Instant::now();
        ddnnf
            .card_of_each_feature(&features_path)
            .unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();

        println!(
            "\nComputed the Cardinality of all features in {} and the results are saved in {}\n\
            It took {} seconds. That is an average of {} seconds per feature",
            file_path,
            features_path,
            elapsed_time,
            elapsed_time / ddnnf.number_of_variables as f64
        );
    }

    // switch in the stream mode
    if cli.stream {
        let stdin_channel = spawn_stdin_channel();
        
        let stdout = io::stdout();
        let mut handle_out = stdout.lock();

        loop {
            match stdin_channel.recv() {
                Ok(mut buffer) => {
                    buffer.pop();

                    let response = ddnnf.handle_stream_msg(&buffer);

                    if response.as_str() == "exit" { handle_out.write_all("ENDE \\ü/\n".as_bytes()).unwrap(); break; }
                    
                    handle_out.write_all(format!("{}\n", response).as_bytes()).unwrap();
                    handle_out.flush().unwrap();
                },
                Err(e) => {
                    handle_out.write_all(format!("error while receiving msg: {}\n", e).as_bytes()).unwrap();
                    handle_out.flush().unwrap();
                },
            }
        }
    }

    // writes the anomalies of the d-DNNF to file
    // anomalies are: core, dead, false-optional features and atomic sets
    if cli.anomalies.is_some() {
        let anomalies_path = build_file_path(cli.anomalies, &file_path, "-anomalies.txt");
        ddnnf.write_anomalies(&anomalies_path).unwrap();
        println!("\nThe anomalies of the d-DNNF (i.e. core, dead, false-optional features, and atomic sets) are written into {}.", anomalies_path);
    }

    // writes the d-DNNF to file
    if cli.save_ddnnf.is_some() {
        let path = build_file_path(cli.save_ddnnf, &file_path, "--saved.nnf");
        write_ddnnf(&mut ddnnf, &path).unwrap();
        println!("\nThe smooth d-DNNF was written into the c2d format in {}.", path);
    }
}

// Uses the supplied file path if there is any.
// If there is no prefix, we switch to the default fallback.
fn build_file_path(maybe_prefix: Option<Vec<String>>, fallback: &String, postfix: &str) -> String {
    let potential_path = maybe_prefix.unwrap();
    let mut custom_file_path;
    if potential_path.is_empty() {
        custom_file_path = fallback.to_owned();
    } else {
        custom_file_path = potential_path[0].clone();
    };
    custom_file_path += postfix;

    custom_file_path
}

// spawns a new thread that listens on stdin and delivers its request to the stream message handling
fn spawn_stdin_channel() -> Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle_in = stdin.lock();
        loop {
            let mut buffer = String::new();
            handle_in.read_line(&mut buffer).unwrap();
            tx.send(buffer).unwrap();
        }
    });
    rx
}