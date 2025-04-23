use anyhow::Result;
use burn::prelude::{Backend, Tensor, TensorData};
use burn::tensor;
use rs_cinic_10_index::images::{RgbImageBatch, load_bhwc_rgbimagebatch};
use rs_cinic_10_index::index::DatasetIndex;
use std::path::Path;

fn batch_to_tensordata(batch: RgbImageBatch) -> TensorData {
    TensorData::from_bytes(batch.data, batch.shape, tensor::DType::U8)
}

pub fn load_bhwc_u8_tensordata_image_batch<P>(paths: &[P]) -> Result<TensorData>
where
    P: AsRef<Path>,
{
    let batch = load_bhwc_rgbimagebatch(paths)?;
    let tensor_data = batch_to_tensordata(batch);
    Ok(tensor_data)
}

pub fn load_bhwc_u8_tensor_image_batch<B, P>(
    paths: &[P],
    device: &B::Device,
) -> Result<Tensor<B, 4>>
where
    B: Backend,
    P: AsRef<Path>,
{
    let data = load_bhwc_u8_tensordata_image_batch(paths)?;
    let tensor = Tensor::from_data(data, device);
    Ok(tensor)
}

pub fn load_hwc_u8_tensor_image<B, P>(
    path: P,
    device: &B::Device,
) -> Result<Tensor<B, 3>>
where
    B: Backend,
    P: AsRef<Path>,
{
    let paths = vec![path.as_ref()];

    let batch = load_bhwc_u8_tensor_image_batch(&paths, device)?;
    let tensor = batch.squeeze(0);

    Ok(tensor)
}

pub trait WithTensorBatches {
    fn load_tensor<B>(
        &self,
        index: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>>
    where
        B: Backend,
    {
        Ok(self.load_tensor_batch(&[index], device)?.squeeze(0))
    }

    fn load_tensor_batch<B>(
        &self,
        indexes: &[usize],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>>
    where
        B: Backend;
}

impl WithTensorBatches for DatasetIndex {
    fn load_tensor_batch<B>(
        &self,
        indexes: &[usize],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>>
    where
        B: Backend,
    {
        let paths = self.indices_to_paths(indexes);
        let tensor = load_bhwc_u8_tensor_image_batch(&paths, device)?;
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    use rs_cinic_10_index::index::{CHANNELS, HEIGHT, ObjectClass, SAMPLES_PER_CLASS, WIDTH};
    use rs_cinic_10_index::{Cinic10Index, default_data_path_or_panic};

    #[test]
    fn test_load_image() -> Result<()> {
        let root_path = default_data_path_or_panic();
        let path = root_path.join("train/airplane/cifar10-train-3318.png");

        let device = Default::default();
        let tensor: Tensor<NdArray, 3> = load_hwc_u8_tensor_image(&path, &device)?;

        assert_eq!(tensor.dims(), [32, 32, 3]);

        Ok(())
    }

    #[test]
    fn test_load_image_batch() -> Result<()> {
        let root_path = default_data_path_or_panic();
        let paths = vec![
            root_path.join("train/airplane/cifar10-train-3318.png"),
            root_path.join("train/airplane/cifar10-train-3318.png"),
        ];

        let device = Default::default();
        let tensor: Tensor<NdArray, 4> = load_bhwc_u8_tensor_image_batch(&paths, &device)?;

        assert_eq!(tensor.dims(), [2, 32, 32, 3]);

        Ok(())
    }

    #[test]
    fn test_load_test_batch() -> Result<()> {
        let cinic: Cinic10Index = Default::default();
        let indices = (0..3).map(|i| i * SAMPLES_PER_CLASS).collect::<Vec<_>>();

        let device = Default::default();
        let tensor: Tensor<NdArray, 4> = cinic.test.load_tensor_batch(&indices, &device)?;
        let classes = cinic.test.indices_to_classes(&indices);

        assert_eq!(tensor.dims(), [3, HEIGHT, WIDTH, CHANNELS]);
        assert_eq!(
            classes,
            vec![
                ObjectClass::Airplane,
                ObjectClass::Automobile,
                ObjectClass::Bird
            ]
        );

        Ok(())
    }
}
