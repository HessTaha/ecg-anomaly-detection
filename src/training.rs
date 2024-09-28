use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::{MseLoss, Reduction};
use burn::{
    optim::AdamConfig, prelude::*, record::NoStdTrainingRecorder, tensor::backend::AutodiffBackend,
};
use burn_core::record::CompactRecorder;
use burn_train::{
    metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
};

use crate::batcher::{EcgBatch, EcgBatcher};
use crate::dataset::EcgDataset;
use crate::model::{AutoEncoder, AutoEncoderConfig, DecoderConfig, EncoderConfig};

static ARTIFACT_DIR: &str = "./tmp/burn-example-ecg";

impl<B: Backend> AutoEncoder<B> {
    pub fn forward_classification(
        &self,
        ecg: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(ecg);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<EcgBatch<B>, RegressionOutput<B>> for AutoEncoder<B> {
    fn step(&self, batch: EcgBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_classification(batch.ecg, batch.ecg_out);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EcgBatch<B>, RegressionOutput<B>> for AutoEncoder<B> {
    fn step(&self, batch: EcgBatch<B>) -> RegressionOutput<B> {
        self.forward_classification(batch.ecg, batch.ecg_out)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(training_config: TrainingConfig, device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);

    let train_dataset = EcgDataset::train();
    let test_dataset = EcgDataset::test();

    let train_batcher = EcgBatcher::<B>::new(device.clone());
    let test_batcher = EcgBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(train_batcher)
        .batch_size(64)
        .shuffle(33)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(test_batcher)
        .shuffle(33)
        .build(test_dataset);

    let encoder_config = EncoderConfig::new(140, 7, 3, 2);
    let decoder_config = DecoderConfig::new(2, 3, 7, 140);

    let autoencoder =
        AutoEncoderConfig::new(encoder_config, decoder_config).init::<B>(&device.clone());

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .devices(vec![device.clone()])
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .summary()
        .build(
            autoencoder,
            training_config.optimizer.init(),
            training_config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    training_config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
