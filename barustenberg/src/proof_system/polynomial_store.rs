use crate::polynomials::Polynomial;
use anyhow::{anyhow, Result};
use ark_ff::{FftField, Field};
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    sync::{Arc, RwLock},
};

#[derive(Debug, Clone, Default)]
pub(crate) struct PolynomialStore<Fr: Field + FftField> {
    polynomial_map: HashMap<String, Arc<RwLock<Polynomial<Fr>>>>,
}

impl<Fr: Field + FftField> PolynomialStore<Fr> {
    pub(crate) fn new() -> Self {
        Self {
            polynomial_map: HashMap::new(),
        }
    }

    /// Transfer ownership of a polynomial to the PolynomialStore
    ///
    /// # Arguments
    /// - `name` - string ID of the polynomial
    /// - `polynomial` - the polynomial to be stored
    pub(crate) fn put(&mut self, name: String, polynomial: Polynomial<Fr>) {
        self.polynomial_map
            .insert(name, Arc::new(RwLock::new(polynomial)));
    }

    pub(crate) fn put_owned(&mut self, name: String, polynomial: Arc<RwLock<Polynomial<Fr>>>) {
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
    pub(crate) fn get(&self, key: &String) -> Result<Arc<RwLock<Polynomial<Fr>>>> {
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
        let poly = Arc::try_unwrap(wrapped_poly).map_err(|_| anyhow!("unwrapping rc failed"))?;
        Ok(poly.into_inner()?)
    }

    /// Get the current size (bytes) of all polynomials in the PolynomialStore
    ///
    /// # Returns
    /// - `usize` - the size
    fn get_size_in_bytes(&self) -> usize {
        let mut size_in_bytes: usize = 0;
        for (_, entry) in self.polynomial_map.iter() {
            size_in_bytes += (**entry).read().unwrap().size() * std::mem::size_of::<Fr>();
        }
        size_in_bytes
    }

    pub(crate) fn insert(&mut self, key: &String, poly: Polynomial<Fr>) {
        self.polynomial_map
            .insert(key.to_string(), Arc::new(RwLock::new(poly)));
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
            let entry_bytes = (**entry).read().unwrap().size() * std::mem::size_of::<Fr>();
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
    fn test_put_then_get() {

        /*
        // Test basic put and get functionality
TEST(PolynomialStore, PutThenGet)
{
    PolynomialStore<Fr> polynomial_store;

    // Instantiate a polynomial with random coefficients
    Polynomial poly(1024);
    for (auto& coeff : poly) {
        coeff = Fr::random_element();
    }

    // Make a copy for comparison after original is moved into container
    Polynomial poly_copy(poly);

    // Move the poly into the container
    polynomial_store.put("id", std::move(poly));

    // Confirm equality of the copy and the original poly that now resides in the container
    EXPECT_EQ(poly_copy, polynomial_store.get("id"));
}
         */
        todo!("PutThenGet")
    }

    #[test]
    fn test_nonexistent_key () {
        /*
        
            PolynomialStore<Fr> polynomial_store;

    polynomial_store.put("id_1", Polynomial(100));

    polynomial_store.get("id_1"); // no problem!

    EXPECT_THROW(polynomial_store.get("id_2"), std::out_of_range);
     */
        todo!("test_nonexistent_key")
    }

    #[test]
    fn test_volume() {
        /*
        // Ensure correct calculation of volume in bytes
TEST(PolynomialStore, Volume)
{
    PolynomialStore<Fr> polynomial_store;
    size_t size1 = 100;
    size_t size2 = 10;
    size_t size3 = 5000;

    Polynomial poly1(size1);
    Polynomial poly2(size2);
    Polynomial poly3(size3);

    polynomial_store.put("id_1", std::move(poly1));
    polynomial_store.put("id_2", std::move(poly2));
    polynomial_store.put("id_3", std::move(poly3));

    // polynomial_store.print();

    size_t bytes_expected = sizeof(Fr) * (size1 + size2 + size3);

    EXPECT_EQ(polynomial_store.get_size_in_bytes(), bytes_expected);
         */

        todo!("test_volume")
    }

    #[test]
    fn test_remove () {
        /*
            PolynomialStore<Fr> polynomial_store;
    size_t size1 = 100;
    size_t size2 = 500;
    Polynomial poly1(size1);
    Polynomial poly2(size2);

    polynomial_store.put("id_1", std::move(poly1));
    polynomial_store.put("id_2", std::move(poly2));

    size_t bytes_expected = sizeof(Fr) * (size1 + size2);

    EXPECT_EQ(polynomial_store.get_size_in_bytes(), bytes_expected);

    polynomial_store.remove("id_1");

    bytes_expected -= sizeof(Fr) * size1;

    EXPECT_THROW(polynomial_store.get("id_1"), std::out_of_range);
    EXPECT_EQ(polynomial_store.get_size_in_bytes(), bytes_expected);
     */
        todo!("test_remove")
    }
}
