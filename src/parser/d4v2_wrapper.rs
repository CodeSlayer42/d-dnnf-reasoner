use std::{process::{Command, self}, fs};

use colored::Colorize;

const D4V2: &[u8] = include_bytes!("d4v2.bin");
const EXECUTABLE_PATH: &str = "src/parser/.d4v2";

/// Using the d4v2 CNF to dDNNF compiler from cril,
/// we take a CNF from path_in and write the dDNNF to path_out
pub(crate) fn compile_cnf(path_in: &str, path_out: &str) {
    // If the byte array is empty, we did not include d4v2. Consequently, we can't compile and have to exit.
    if D4V2.is_empty() {
        eprintln!("{}", "\nERROR: d4v2 is not part of that binary! \
            Hence, CNF files can not be handled by this binary file. \
            Compile again with 'EXCLUDE_D4V2=FALSE cargo build --release' to use d4v2.\nAborting...".red());
        process::exit(1);
    }

    // persist the binary data to a callable file
    std::fs::write(EXECUTABLE_PATH, D4V2)
        .expect("failed to write file");
    set_permissions();
    
    // execute the command to compile a dDNNF from a CNF file
    Command::new(EXECUTABLE_PATH)
        .args(["-i", path_in, "-m", "ddnnf-compiler", "--dump-ddnnf", path_out])
        .output().unwrap();
    
    // Remove it again, after the call. This operation is very cheap!
    fs::remove_file(EXECUTABLE_PATH).unwrap();
}

// When writing the stored bytes of the binary to a file, 
// we have to adjust the permissions of that file to execute commands on that binary.
fn set_permissions() {
    // Set executable permissions on Unix systems
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(EXECUTABLE_PATH)
            .expect("failed to get metadata")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(EXECUTABLE_PATH, perms)
            .expect("failed to set permissions");
    }

    // Set executable permissions on Windows systems
    #[cfg(windows)]
    {
        use std::fs::OpenOptions;
        use std::os::windows::fs::OpenOptionsExt;
        let mut options = OpenOptions::new();
        options.write(true)
            .create(true)
            .truncate(true)
            .custom_flags(winapi::um::winnt::FILE_ATTRIBUTE_NORMAL | winapi::um::winnt::FILE_FLAG_BACKUP_SEMANTICS);
        let file = options.open(EXECUTABLE_PATH)
            .expect("failed to open file");
        let mut permissions = file.metadata()
            .expect("failed to get metadata")
            .permissions();
        permissions.set_readonly(false);
        file.set_permissions(permissions)
            .expect("failed to set permissions");
    }
}