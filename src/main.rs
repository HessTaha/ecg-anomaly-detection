mod batcher;
mod dataset;
mod model;
mod training;

use burn::backend::Wgpu;
use burn::optim::AdamConfig;
use burn_autodiff::Autodiff;

use crate::training::{train, TrainingConfig};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // Create a default Wgpu device
    let device = burn::backend::wgpu::WgpuDevice::default();
    let optim_config = AdamConfig::new();

    train::<MyAutodiffBackend>(TrainingConfig::new(optim_config), device.clone());
}
