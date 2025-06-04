use anyhow::Result;
use image::RgbImage;
use std::path::Path;

/// Loads an RGB image from the given path.
///
/// # Parameters
///
/// - `path`: The path to the image file.
///
/// # Returns
///
/// A result containing the loaded image.
pub fn load_rgbimage<P>(path: P) -> Result<RgbImage>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let img = image::open(path)?;

    /*
    let color_type = img.color();
    assert_eq!(
        color_type,
        ColorType::Rgb8,
        "Image color type must be Rgb8, found: {:?}",
        color_type
    );
     */

    Ok(img.to_rgb8())
}

/// A structure representing a batch of RGB images.
#[derive(Debug, Clone)]
pub struct RgbImageBatch {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
}

impl RgbImageBatch {
    /// Creates a new `RgbImageBatch` with the given shape.
    ///
    /// # Parameters
    ///
    /// - `shape`: A slice representing the shape of the batch.
    ///
    /// # Returns
    ///
    /// A new `RgbImageBatch` instance.
    pub fn new(shape: &[usize]) -> Self {
        let shape = shape.to_vec();
        let size = shape.iter().product();
        let data = Vec::with_capacity(size);

        assert_eq!(shape.len(), 4);
        assert_eq!(shape[3], 3);

        Self { shape, data }
    }

    /// Pushes RGB pixel data into the batch.
    ///
    /// # Parameters
    ///
    /// - `img`: A reference to the RGB image to be pushed.
    ///
    /// # Returns
    ///
    /// None
    pub(crate) fn push_rgb_pixels(
        &mut self,
        img: &RgbImage,
    ) {
        for rgb in img.pixels() {
            self.data.push(rgb[0]);
            self.data.push(rgb[1]);
            self.data.push(rgb[2]);
        }
    }
    pub fn batch_size(&self) -> usize {
        self.shape[0]
    }

    pub fn height(&self) -> usize {
        self.shape[1]
    }

    pub fn width(&self) -> usize {
        self.shape[2]
    }

    pub fn channels(&self) -> usize {
        self.shape[3]
    }

    pub fn size(&self) -> usize {
        self.data.capacity()
    }
}

/// Loads a batch of images from the given paths.
///
/// The function takes a slice of paths, a function to create the batch dimensions,
/// and a function to process each image.
///
/// # Parameters
///
/// - `paths`: A slice of paths to the images.
/// - `on_dims`: Called with dimensions, to build the batch object.
/// - `on_img`: Called for each loaded image.
///
/// # Returns
///
/// A result containing the batch of images.
pub fn load_batch<T, P>(
    paths: &[P],
    on_dims: fn(&[usize; 4]) -> Result<T>,
    on_img: fn(&mut T, idx: usize, img: &RgbImage) -> Result<()>,
) -> Result<T>
where
    P: AsRef<Path>,
{
    let batch_size = paths.len();

    let path = paths.first().unwrap().as_ref();
    let img = load_rgbimage(path)?;

    let (width, height) = img.dimensions();
    let shape = [batch_size, height as usize, width as usize, 3];

    let mut batch = on_dims(&shape)?;
    on_img(&mut batch, 0, &img)?;

    for i in 1..batch_size {
        let path = paths.get(i).unwrap();
        let img = load_rgbimage(path)?;

        assert_eq!(
            img.dimensions(),
            (width, height),
            "Image dimensions do not match"
        );
        on_img(&mut batch, i, &img)?;
    }

    Ok(batch)
}

/// Loads a batch of RGB images from the given paths into a single `RgbImageBatch`.
///
/// # Parameters
///
/// - `paths`: A slice of paths to the images.
///
/// # Returns
///
/// A result containing the batch of images.
pub fn load_bhwc_rgbimagebatch<P>(paths: &[P]) -> Result<RgbImageBatch>
where
    P: AsRef<Path>,
{
    load_batch::<RgbImageBatch, _>(
        paths,
        |shape| Ok(RgbImageBatch::new(shape)),
        |batch, _idx, img| {
            batch.push_rgb_pixels(img);
            Ok(())
        },
    )
}
