use crate::polynomials::Polynomial;
use anyhow::{anyhow, Result};
use ark_ff::{FftField, Field};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    rc::Rc,
};

#[derive(Debug, Clone, Default)]
pub(crate) struct PolynomialStore<Fr: Field + FftField> {
    polynomial_map: HashMap<String, Rc<RefCell<Polynomial<Fr>>>>,
    phantom: PhantomData<Fr>,
}

impl<Fr: Field + FftField> PolynomialStore<Fr> {
    pub(crate) fn new() -> Self {
        Self {
            polynomial_map: HashMap::new(),
            phantom: PhantomData,
        }
    }

    /// Transfer ownership of a polynomial to the PolynomialStore
    ///
    /// # Arguments
    /// - `name` - string ID of the polynomial
    /// - `polynomial` - the polynomial to be stored
    pub(crate) fn put(&mut self, name: String, polynomial: Polynomial<Fr>) {
        self.polynomial_map
            .insert(name, Rc::new(RefCell::new(polynomial)));
    }

    /// Get a reference to a polynomial in the PolynomialStore; will throw exception if the
    /// key does not exist in the map
    ///
    /// # Arguments
    /// - `key` - string ID of the polynomial
    ///
    /// # Returns
    /// - `Result<Polynomial>` - a reference to the polynomial associated with the given key
    pub(crate) fn get(&self, key: &String) -> Result<Rc<RefCell<Polynomial<Fr>>>> {
        self.polynomial_map
            .get(key)
            .ok_or_else(|| anyhow!("didn't find polynomial..."))
            .map(|a| a.clone())
    }

    /// Erase a polynomial from the PolynomialStore; will throw exception if the key does not exist
    ///
    /// # Arguments
    /// - `key` - string ID of the polynomial
    ///
    /// # Returns
    /// - `Result<Polynomial>` - the polynomial associated with the given key
    pub(crate) fn remove(&mut self, key: String) -> Result<Polynomial<Fr>> {
        let wrapped_poly = self
            .polynomial_map
            .remove(&key)
            .ok_or_else(|| anyhow!("didn't find polynomial..."))?;
        let poly = Rc::try_unwrap(wrapped_poly).map_err(|_| anyhow!("unwrapping rc failed"))?;
        Ok(poly.into_inner())
    }

    /// Get the current size (bytes) of all polynomials in the PolynomialStore
    ///
    /// # Returns
    /// - `usize` - the size
    fn get_size_in_bytes(&self) -> usize {
        let mut size_in_bytes: usize = 0;
        for (_, entry) in self.polynomial_map.iter() {
            size_in_bytes += entry.borrow().size() * std::mem::size_of::<Fr>();
        }
        size_in_bytes
    }

    pub(crate) fn insert(&mut self, key: &String, poly: Polynomial<Fr>) {
        self.polynomial_map
            .insert(key.to_string(), Rc::new(RefCell::new(poly)));
    }

    fn contains(&self, key: &String) -> bool {
        self.polynomial_map.contains_key(key)
    }

    fn len(&self) -> usize {
        self.polynomial_map.len()
    }
    // TODO: "allow for const range based for loop"
}

impl<Fr: Field + FftField> Display for PolynomialStore<Fr> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let size_in_mb = (self.get_size_in_bytes() / 1_000_000) as f32;
        write!(f, "PolynomialStore contents total size: {} MB", size_in_mb)?;
        for (key, entry) in self.polynomial_map.iter() {
            let entry_bytes = entry.borrow().size() * std::mem::size_of::<Fr>();
            write!(
                f,
                "PolynomialStore: {} -> {} bytes, {:?}",
                key, entry_bytes, entry
            )?;
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
