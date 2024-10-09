# Ecg anomaly detection

This is a learning project with the purpose of experimenting machine learning project with rust 😁. 
The ecg anomaly detection is a well known learning resources for autoencoders models.
The data can be found ~> [here](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000)

## Setup and install 

1. Pre requisite

- Rust / rustup
- Cargo
- Download and extract the data to a local folder 
- Update the `dataset.rs`file with the path to your data ~> [here](./src/dataset.rs#L46)

2. Build the project

```shell
cargo build
```

3. Run and view training dashboard 
 
```shell
cargo run
```