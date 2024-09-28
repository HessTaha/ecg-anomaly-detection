use burn::data::dataset::{transform::PartialDataset, Dataset, InMemDataset};
use csv;

pub type EcgRawRecord = Vec<f32>;
pub type PartialEcg = PartialDataset<InMemDataset<EcgRawRecord>, EcgRawRecord>;

pub fn read_ecg_raw_record(path: &str) -> Vec<EcgRawRecord> {
    let mut rdr = csv::ReaderBuilder::new();
    let rdr = rdr.delimiter(b',');

    let mut _data = rdr.from_path(path).expect("Could not read the file");

    let mut ecg_raw_data = Vec::new();

    for record_option in _data.deserialize() {
        let record: Vec<f32> = record_option.unwrap();
        let record: EcgRawRecord = record[..140].try_into().unwrap();
        ecg_raw_data.push(record);
    }
    ecg_raw_data
}

pub struct EcgDataset {
    pub dataset: PartialEcg,
}

impl EcgDataset {
    pub fn new(split: &str, path: &str) -> Self {
        let raw: Vec<EcgRawRecord> = read_ecg_raw_record(&path);

        let dataset = InMemDataset::new(raw);
        let len = dataset.len();

        let filtered_dataset = match split {
            "train" => PartialEcg::new(dataset, 0, len * 8 / 10),
            "test" => PartialEcg::new(dataset, len * 8 / 10, len),
            _ => panic!("Invalid split type"),
        };

        Self {
            dataset: filtered_dataset,
        }
    }

    pub fn train() -> Self {
        Self::new("train", "./src/data/ecg.csv")
    }
    pub fn test() -> Self {
        Self::new("test", "./src/data/ecg.csv")
    }
}

impl Dataset<EcgRawRecord> for EcgDataset {
    fn get(&self, index: usize) -> Option<EcgRawRecord> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
