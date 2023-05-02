use crate::{ecc::Field, polynomials::Polynomial};
use anyhow::{anyhow, Result};
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};

#[derive(Debug, Clone)]
pub(crate) struct PolynomialStore<Fr: Field> {
    polynomial_map: HashMap<String, Polynomial<Fr>>,
}
impl<Fr: Field> PolynomialStore<Fr> {
    pub(crate) const fn new() -> Self {
        Self {
            polynomial_map: HashMap::new(),
        }
    }

    /// Transfer ownership of a polynomial to the PolynomialStore
    ///
    /// # Arguments
    /// - `name` - string ID of the polynomial
    /// - `polynomial` - the polynomial to be stored
    pub(crate) const fn put(&mut self, name: String, polynomial: Polynomial<Fr>) {
        self.polynomial_map.insert(name, polynomial);
    }

    /// Get a reference to a polynomial in the PolynomialStore; will throw exception if the
    /// key does not exist in the map
    ///
    /// # Arguments
    /// - `key` - string ID of the polynomial
    ///
    /// # Returns
    /// - `Result<Polynomial>` - a reference to the polynomial associated with the given key
    pub(crate) const fn get(&self, key: String) -> Result<Polynomial<Fr>> {
        self.polynomial_map
            .get(&key)
            .ok_or(anyhow!("didn't find polynomial..."))
            .cloned()
    }

    /// Erase a polynomial from the PolynomialStore; will throw exception if the key does not exist
    ///
    /// # Arguments
    /// - `key` - string ID of the polynomial
    ///
    /// # Returns
    /// - `Result<Polynomial>` - the polynomial associated with the given key
    pub(crate) const fn remove(&mut self, key: String) -> Result<Polynomial<Fr>> {
        self.polynomial_map
            .remove(&key)
            .ok_or(anyhow!("didn't find polynomial..."))
    }

    /// Get the current size (bytes) of all polynomials in the PolynomialStore
    ///
    /// # Returns
    /// - `usize` - the size
    fn get_size_in_bytes(&self) -> usize {
        let mut size_in_bytes: usize = 0;
        for (_, entry) in self.polynomial_map.iter() {
            size_in_bytes += entry.size() * Fr::SizeInBytes::to_int();
        }
        size_in_bytes
    }

    fn contains(&self, key: String) -> bool {
        self.polynomial_map.contains_key(&key)
    }

    fn len(&self) -> usize {
        self.len()
    }

    // TODO: "allow for const range based for loop"
}

impl<Fr: Field> Display for PolynomialStore<Fr> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let size_in_mb = (self.get_size_in_bytes() / 1e6) as f32;
        write!(f, "PolynomialStore contents total size: {} MB", size_in_mb);
        for (key, entry) in self.polynomial_map.iter() {
            let entry_bytes = entry.size() * Fr::SizeInBytes::to_int();
            write!(
                f,
                "PolynomialStore: {} -> {} bytes, {:?}",
                key, entry_bytes, entry
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_todo() {
        todo!("check out polynomial_store.test.cpp")
    }
}
