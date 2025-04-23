use crate::default_data_path_or_panic;
use crate::images::{RgbImageBatch, load_bhwc_rgbimagebatch};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fs, io};
use strum::{EnumCount, IntoEnumIterator};

pub const HEIGHT: usize = 32;
pub const WIDTH: usize = 32;
pub const CHANNELS: usize = 3;

pub const CONTRIB_FILE: &str = "imagenet-contributors.csv";
pub const SYNSET_FILE: &str = "synsets-to-cifar-10-classes.txt";

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum_macros::EnumString,
    strum_macros::Display,
    strum_macros::EnumIter,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum DataSet {
    Train,
    Test,
    Valid,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum_macros::Display,
    strum_macros::EnumIter,
    strum_macros::EnumCount,
    strum_macros::EnumString,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ObjectClass {
    Airplane,
    Automobile,
    Bird,
    Cat,
    Deer,
    Dog,
    Frog,
    Horse,
    Ship,
    Truck,
}

pub const SAMPLES_PER_CLASS: usize = 9000;
pub const SAMPLES_PER_DATASET: usize = SAMPLES_PER_CLASS * ObjectClass::COUNT;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexRecord {
    synset: String,
    image_num: usize,
    data_set: DataSet,
    class: ObjectClass,
}

impl TryFrom<&csv::StringRecord> for IndexRecord {
    type Error = csv::Error;

    fn try_from(row: &csv::StringRecord) -> Result<Self, Self::Error> {
        row.deserialize(None)
    }
}

/// Parse the `CONTRIB_FILE` CSV file into a vector of IndexRecord.
///
/// The CSV file is expected to have the following columns:
/// - `synset`: Synset ID
/// - `image_num`: Image number
/// - `cinic_set`: Dataset (train, test, valid)
/// - `class`: CIFAR Object class
///
/// # Parameters:
///
/// - `rdr`: A reader that implements the `io::Read` trait.
///
/// # Returns:
///
/// A `Result` containing a vector of `IndexRecord` on success, or an error on failure.
pub fn parse_contrib_index<R>(rdr: R) -> Result<Vec<IndexRecord>>
where
    R: io::Read,
{
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .trim(csv::Trim::All)
        .from_reader(rdr);

    let records = rdr
        .records()
        .map(|r| r.unwrap().deserialize(None).unwrap())
        .collect::<Vec<IndexRecord>>();

    Ok(records)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SynsetNode {
    pub synset_id: String,
    pub object_class: ObjectClass,
    pub synset_base_id: Option<String>,
    pub aliases: Vec<String>,
}

/// Parse the `SYNSET_FILE` file into a HashMap of SynsetNode.
///
/// # Parameters:
///
/// - `rdr`: A reader that implements the `io::Read` trait.
///
/// # Returns:
///
/// A `Result` containing a HashMap of SynsetNode on success, or an error on failure.
pub fn parse_synset_map<R>(rdr: R) -> Result<HashMap<String, SynsetNode>>
where
    R: io::Read,
{
    let mut synset_map: HashMap<String, SynsetNode> = HashMap::new();

    let mut object_class: Option<ObjectClass> = None;
    let mut synset_stack: Vec<(usize, String)> = Vec::new();

    let rdr = io::BufReader::new(rdr);
    for res in rdr.lines() {
        let full_line = res?;
        let cur = full_line.trim();

        if !cur.starts_with("-") {
            object_class = Some(ObjectClass::from_str(cur)?);
            synset_stack.clear();
            continue;
        }

        assert!(
            object_class.is_some(),
            "Malformed synset map; synset entry before class: {}",
            full_line
        );

        // This line is a nested synset node:
        //   (?P<depth>: "-"+)
        //   (?P<synset>: n[0-9]+)
        //   ":"
        //   {comma separated list of names}
        //
        // Ex:
        //    "----------------n123: alias, alias two, banana"
        //    "--------------------n456: more specific set, child of n123"

        // Count the "-" characters at the head of the line.
        let orig_len = cur.len();
        let line: &str = cur.trim_start_matches('-');
        let depth = orig_len - line.len();

        let split_pos = line.find(':').unwrap();
        let synset_id = line[..split_pos].to_owned();

        let aliases = &line[split_pos + 1..];
        let aliases: Vec<String> = aliases.split(',').map(|s| s.trim().to_string()).collect();

        // Remove all nodes too deep to be our parent.
        while let Some((d, _)) = synset_stack.last() {
            if *d < depth {
                break;
            }
            synset_stack.pop();
        }

        // If a parent exists, get the id.
        let synset_base_id = synset_stack
            .last()
            .map(|(_, parent_id)| parent_id.to_string());
        // Add this node to the stack.
        synset_stack.push((depth, synset_id.clone()));

        let node = SynsetNode {
            synset_id,
            object_class: object_class.unwrap(),
            synset_base_id,
            aliases,
        };

        synset_map.insert(node.synset_id.clone(), node);
    }

    Ok(synset_map)
}

/// Lists all PNG files in the given directory, sorted by their names.
///
/// # Parameters
///
/// - `dir`: The directory to list the PNG files from.
///
/// # Returns
///
/// A result containing a vector of paths to the PNG files.
fn list_pngs_sorted<P>(dir: P) -> Result<Vec<PathBuf>>
where
    P: AsRef<Path>,
{
    let dir = dir.as_ref();

    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if let Some(ext) = entry.path().extension() {
                    if ext == "png" {
                        return Some(entry.path().to_str().unwrap().to_string());
                    }
                }
            }
            None
        })
        .map(PathBuf::from)
        .collect();

    files.sort();

    Ok(files)
}

#[derive(Debug, Clone)]
pub struct DatasetIndex {
    ds_path: PathBuf,
    items: Vec<(ObjectClass, PathBuf)>,
}

impl DatasetIndex {
    fn load_index_from_dir(ds_path: &Path) -> Result<Self> {
        let ds_path = ds_path.to_path_buf();
        let mut items = Vec::with_capacity(SAMPLES_PER_DATASET);

        for oc in ObjectClass::iter() {
            let oc_path = ds_path.join(oc.to_string());
            items.extend(list_pngs_sorted(&oc_path)?.into_iter().map(|p| (oc, p)))
        }

        let di = Self { ds_path, items };
        assert_eq!(di.len(), SAMPLES_PER_DATASET);

        Ok(di)
    }

    pub fn ds_path(&self) -> &Path {
        &self.ds_path
    }

    /// Get the size of the dataset.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Convert an item index to an object class.
    pub fn index_to_class(
        &self,
        index: usize,
    ) -> ObjectClass {
        self.items[index].0
    }

    /// Convert a slice of indices to a vector of object classes.
    pub fn indices_to_classes(
        &self,
        indices: &[usize],
    ) -> Vec<ObjectClass> {
        indices.iter().map(|&i| self.index_to_class(i)).collect()
    }

    /// Convert an item index to an image path.
    ///
    /// # Parameters
    ///
    /// - `index`: The index of the item.
    ///
    /// # Returns
    ///
    /// A `PathBuf` representing the path to the image.
    pub fn index_to_path(
        &self,
        index: usize,
    ) -> PathBuf {
        let (oc, fname) = &self.items[index];
        self.ds_path.join(oc.to_string()).join(fname)
    }

    /// Convert a slice of indices to a vector of image paths.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices to convert.
    ///
    /// # Returns
    ///
    /// A vector of `PathBuf` representing the paths to the images.
    pub fn indices_to_paths(
        &self,
        indices: &[usize],
    ) -> Vec<PathBuf> {
        indices.iter().map(|&i| self.index_to_path(i)).collect()
    }

    /// Load an `RgbImageBatch` for a batch of indexes in the dataset.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices to load.
    ///
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded `RgbImageBatch` on success, or an error on failure.
    pub fn load_rgbimagebatch(
        &self,
        indices: &[usize],
    ) -> Result<RgbImageBatch> {
        let paths = self.indices_to_paths(indices);
        load_bhwc_rgbimagebatch(&paths)
    }
}

/// The main index for the CINIC-10 dataset.
#[derive(Debug, Clone)]
pub struct Cinic10Index {
    pub root: PathBuf,

    pub imagenet_contrib: Vec<IndexRecord>,
    pub synset_map: HashMap<String, SynsetNode>,

    pub train: DatasetIndex,
    pub test: DatasetIndex,
    pub valid: DatasetIndex,
}

impl Cinic10Index {
    /// Create a new `Cinic10Index` from the given directory.
    ///
    /// # Parameters
    ///
    /// - `root`: The root directory of the CINIC-10 dataset.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Cinic10Index` on success, or an error on failure.
    pub fn new_from_dir<P>(root: P) -> Result<Cinic10Index>
    where
        P: AsRef<Path>,
    {
        let root = root.as_ref();

        if !root.exists() {
            panic!("CINIC-10 dataset not found at {}", root.display());
        }
        if !root.is_dir() {
            panic!(
                "CINIC-10 dataset path is not a directory: {}",
                root.display()
            );
        }

        let index = parse_contrib_index(File::open(root.join(CONTRIB_FILE))?)?;

        Ok(Cinic10Index {
            root: root.to_path_buf(),
            imagenet_contrib: index,
            synset_map: parse_synset_map(File::open(root.join(SYNSET_FILE))?)?,
            train: DatasetIndex::load_index_from_dir(&root.join(DataSet::Train.to_string()))?,
            test: DatasetIndex::load_index_from_dir(&root.join(DataSet::Test.to_string()))?,
            valid: DatasetIndex::load_index_from_dir(&root.join(DataSet::Valid.to_string()))?,
        })
    }
}

impl Default for Cinic10Index {
    /// Create a new Cinic10Index with the current default path.
    ///
    /// # Returns
    ///
    /// A new `Cinic10Index` instance.
    ///
    /// # Panics
    ///
    /// Panics if the default path does not exist, is not a directory,
    /// or does not contain a valid CINIC-10 dataset.
    fn default() -> Self {
        Cinic10Index::new_from_dir(default_data_path_or_panic()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use csv::StringRecord;
    use indoc::{formatdoc, indoc};

    use std::str::FromStr;

    #[test]
    fn parse_dataset() {
        assert_eq!(DataSet::from_str("train").unwrap(), DataSet::Train);
        assert_eq!(DataSet::from_str("test").unwrap(), DataSet::Test);
        assert_eq!(DataSet::from_str("valid").unwrap(), DataSet::Valid);

        assert!(DataSet::from_str("bad").is_err());
        assert!(DataSet::from_str("").is_err());
    }

    #[test]
    fn parse_class() {
        assert_eq!(
            ObjectClass::from_str("airplane").unwrap(),
            ObjectClass::Airplane
        );
        assert_eq!(
            ObjectClass::from_str("automobile").unwrap(),
            ObjectClass::Automobile
        );
        assert_eq!(ObjectClass::from_str("bird").unwrap(), ObjectClass::Bird);
        assert_eq!(ObjectClass::from_str("cat").unwrap(), ObjectClass::Cat);
        assert_eq!(ObjectClass::from_str("deer").unwrap(), ObjectClass::Deer);
        assert_eq!(ObjectClass::from_str("dog").unwrap(), ObjectClass::Dog);
        assert_eq!(ObjectClass::from_str("frog").unwrap(), ObjectClass::Frog);
        assert_eq!(ObjectClass::from_str("horse").unwrap(), ObjectClass::Horse);
        assert_eq!(ObjectClass::from_str("ship").unwrap(), ObjectClass::Ship);
        assert_eq!(ObjectClass::from_str("truck").unwrap(), ObjectClass::Truck);

        assert!(ObjectClass::from_str("bad").is_err());
        assert!(ObjectClass::from_str("").is_err());
    }

    #[test]
    fn parse_index_record() {
        let expected = IndexRecord {
            synset: "n02123045".to_string(),
            image_num: 1,
            data_set: DataSet::Train,
            class: ObjectClass::Airplane,
        };

        let row = StringRecord::from(vec!["n02123045", "1", "train", "airplane"]);
        let record: IndexRecord = IndexRecord::try_from(&row).unwrap();

        assert_eq!(record, expected);
    }

    #[test]
    fn test_parse_index_from_reader() {
        let source = indoc! {"
            synset, image_num, cinic_set, class
            n02704645, 14894, train, airplane
            n02690373, 6332, valid, frog
        "};

        let rdr = io::Cursor::new(source);
        let records = parse_contrib_index(rdr).unwrap();

        assert_eq!(records.len(), 2);

        let record = records.first().unwrap();
        assert_eq!(record.synset, "n02704645");
        assert_eq!(record.image_num, 14894);
        assert_eq!(record.data_set, DataSet::Train);
        assert_eq!(record.class, ObjectClass::Airplane);

        let record = records.get(1).unwrap();
        assert_eq!(record.synset, "n02690373");
        assert_eq!(record.image_num, 6332);
        assert_eq!(record.data_set, DataSet::Valid);
        assert_eq!(record.class, ObjectClass::Frog);
    }

    #[test]
    fn test_parse_synsets_from_reader() -> Result<()> {
        let source = formatdoc! {"
            {dog}
            ----n123: good boy
            ------n1230: bestest boy
            ----n9: cujo
            {cat}
            ----n999: chonk
            ",
            dog = ObjectClass::Dog,
            cat = ObjectClass::Cat,
        };

        let rdr = std::io::Cursor::new(source);
        let synset_map = parse_synset_map(rdr)?;

        assert_eq!(
            synset_map.get("n123").unwrap(),
            &SynsetNode {
                synset_id: "n123".to_string(),
                object_class: ObjectClass::Dog,
                synset_base_id: None,
                aliases: vec!["good boy".to_string()],
            }
        );

        assert_eq!(
            synset_map.get("n1230").unwrap(),
            &SynsetNode {
                synset_id: "n1230".to_string(),
                object_class: ObjectClass::Dog,
                synset_base_id: Some("n123".to_string()),
                aliases: vec!["bestest boy".to_string()],
            }
        );

        assert_eq!(
            synset_map.get("n9").unwrap(),
            &SynsetNode {
                synset_id: "n9".to_string(),
                object_class: ObjectClass::Dog,
                synset_base_id: None,
                aliases: vec!["cujo".to_string()],
            }
        );

        assert_eq!(
            synset_map.get("n999").unwrap(),
            &SynsetNode {
                synset_id: "n999".to_string(),
                object_class: ObjectClass::Cat,
                synset_base_id: None,
                aliases: vec!["chonk".to_string()],
            }
        );

        Ok(())
    }

    #[test]
    fn test_load_test_batch() -> Result<()> {
        let cinic: Cinic10Index = Default::default();
        let indices = (0..3).map(|i| i * SAMPLES_PER_CLASS).collect::<Vec<_>>();

        let batch = cinic.test.load_rgbimagebatch(&indices)?;
        let classes = cinic.test.indices_to_classes(&indices);

        assert_eq!(batch.shape, [3, HEIGHT, WIDTH, CHANNELS]);
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
