use burn::{
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    pub input_size: usize,
    pub linear1_output_dim: usize,
    pub linear2_output_dim: usize,
    pub linear3_output_dim: usize,
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    pub input_size: usize,
    pub linear1_output_dim: usize,
    pub linear2_output_dim: usize,
    pub linear3_output_dim: usize,
}

#[derive(Module, Debug)]
pub struct AutoEncoder<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    linear_activation: Relu,
    output_activation: Sigmoid,
}

#[derive(Config, Debug)]
pub struct AutoEncoderConfig {
    encoder_config: EncoderConfig,
    decoder_config: DecoderConfig,
}

impl AutoEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoEncoder<B> {
        let encoder = Encoder {
            linear1: LinearConfig::new(
                self.encoder_config.input_size,
                self.encoder_config.linear1_output_dim,
            )
            .init(device),
            linear2: LinearConfig::new(
                self.encoder_config.linear1_output_dim,
                self.encoder_config.linear2_output_dim,
            )
            .init(device),
            linear3: LinearConfig::new(
                self.encoder_config.linear2_output_dim,
                self.encoder_config.linear3_output_dim,
            )
            .init(device),
        };
        let decoder = Decoder {
            linear1: LinearConfig::new(
                self.decoder_config.input_size,
                self.decoder_config.linear1_output_dim,
            )
            .init(device),
            linear2: LinearConfig::new(
                self.decoder_config.linear1_output_dim,
                self.decoder_config.linear2_output_dim,
            )
            .init(device),
            linear3: LinearConfig::new(
                self.decoder_config.linear2_output_dim,
                self.decoder_config.linear3_output_dim,
            )
            .init(device),
        };

        AutoEncoder {
            encoder: encoder,
            decoder: decoder,
            linear_activation: Relu::new(),
            output_activation: Sigmoid::new(),
        }
    }
}

impl<B: Backend> AutoEncoder<B> {
    pub fn forward(&self, ecg: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.encoder.linear1.forward(ecg);
        let x = self.linear_activation.forward(x);
        let x = self.encoder.linear2.forward(x);
        let x = self.linear_activation.forward(x);
        let x = self.encoder.linear3.forward(x);
        let x = self.linear_activation.forward(x);

        let x = self.decoder.linear1.forward(x);
        let x = self.linear_activation.forward(x);
        let x = self.decoder.linear2.forward(x);
        let x = self.linear_activation.forward(x);
        let x = self.decoder.linear3.forward(x);
        let x = self.linear_activation.forward(x);

        self.output_activation.forward(x)
    }
}
