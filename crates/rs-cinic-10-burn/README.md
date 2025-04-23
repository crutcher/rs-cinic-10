# Setup

## 1. Download / Install CINIC-10

Download the CINIC-10 dataset from the official website:
[CINIC-10](https://github.com/BayesWatch/cinic-10)

Extract the downloaded dataset to a directory of your choice.

## 2. Configure CINIC10_PATH

There are two options to configure the path to the CINIC-10 dataset:

1. **Environment Variable**: Set the `CINIC10_PATH` environment variable to the path of the extracted CINIC-10 dataset. This can be done in your terminal or command prompt:
   ```bash
   export CINIC10_PATH=/path/to/cinic-10
   ```
   
2. **Setup Code**: Alternatively, you can set the default path:
   ```rust
   use cinic_10_index::set_default_path;
   
   fn main() {
    set_default_path("/path/to/cinic-10");
   
    // ...
   }
   ```
   
## 3. Load default Index

You can load the default index using the following code:
```rust
use cinic_10_index::Cinic10Index;

fn main() {
   let index: Cinic10Index = Default::default();
   ...
}
```
