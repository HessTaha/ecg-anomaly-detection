use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use crate::dataset::EcgRawRecord;

#[derive(Clone)]
pub struct EcgBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct EcgBatch<B: Backend> {
    pub ecg: Tensor<B, 2>,
    pub ecg_out: Tensor<B, 2>,
}

impl<B: Backend> EcgBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<EcgRawRecord, EcgBatch<B>> for EcgBatcher<B> {
    fn batch(&self, items: Vec<EcgRawRecord>) -> EcgBatch<B> {
        let flattened: Vec<f32> = items.iter().flatten().copied().collect();

        let batch_size = items.len();
        let sequence_length = items[0].len();
        let ecg_shape = [batch_size, sequence_length];

        let x_tensor = TensorData::new(flattened, ecg_shape);
        let y_tensor = x_tensor.clone();

        let x = Tensor::from_data(x_tensor, &self.device);
        let y = Tensor::from_data(y_tensor, &self.device);

        EcgBatch { ecg: x, ecg_out: y }
    }
}
