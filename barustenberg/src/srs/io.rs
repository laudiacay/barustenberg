use anyhow::{anyhow, Result};
use ark_bn254::Fq;
use ark_bn254::Fq2;
use ark_bn254::{G1Affine, G2Affine};
use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;
use std::cmp::min;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;

const BLAKE2B_CHECKSUM_LENGTH: usize = 64;

#[derive(Debug, Default)]
struct Manifest {
    transcript_number: u32,
    total_transcripts: u32,
    total_g1_points: u32,
    total_g2_points: u32,
    num_g1_points: u32,
    num_g2_points: u32,
    start_from: u32,
}

fn get_transcript_size(manifest: &Manifest) -> usize {
    let manifest_size = std::mem::size_of::<Manifest>();
    let g1_buffer_size = std::mem::size_of::<Fq>() * 2 * manifest.num_g1_points as usize;
    let g2_buffer_size = std::mem::size_of::<Fq2>() * 2 * manifest.num_g2_points as usize;
    manifest_size + g1_buffer_size + g2_buffer_size + BLAKE2B_CHECKSUM_LENGTH
}

fn read_manifest(filename: &str) -> Result<Manifest> {
    let mut file = File::open(filename)?;
    let mut manifest = Manifest::default(); // Make sure your Manifest struct implements Default

    // Here you need to call file.read_u32::<BigEndian>()? for each field in Manifest
    manifest.transcript_number = file.read_u32::<BigEndian>()?;
    manifest.total_transcripts = file.read_u32::<BigEndian>()?;
    manifest.total_g1_points = file.read_u32::<BigEndian>()?;
    manifest.total_g2_points = file.read_u32::<BigEndian>()?;
    manifest.num_g1_points = file.read_u32::<BigEndian>()?;
    manifest.num_g2_points = file.read_u32::<BigEndian>()?;
    manifest.start_from = file.read_u32::<BigEndian>()?;

    Ok(manifest)
}

fn write_manifest(filename: &str, manifest: &Manifest) -> Result<()> {
    let mut file = File::create(filename)?;

    // Here you need to call file.write_u32::<BigEndian>(value)? for each field in Manifest
    file.write_u32::<BigEndian>(manifest.transcript_number)?;
    file.write_u32::<BigEndian>(manifest.total_transcripts)?;
    file.write_u32::<BigEndian>(manifest.total_g1_points)?;
    file.write_u32::<BigEndian>(manifest.total_g2_points)?;
    file.write_u32::<BigEndian>(manifest.num_g1_points)?;
    file.write_u32::<BigEndian>(manifest.num_g2_points)?;
    file.write_u32::<BigEndian>(manifest.start_from)?;

    Ok(())
}

fn read_g1_elements_from_buffer(elements: &mut [G1Affine], buffer: &[u8]) {
    elements.copy_from_slice(&buffer);
    byteswap_g1(elements);
}

fn byteswap_g1(elements: &mut [G1Affine]) {
    let num_elements = elements.len();

    if cfg!(target_endian = "little") {
        for element in elements.iter_mut() {
            element.x.data = element.x.data.to_be_bytes();
            element.y.data = element.y.data.to_be_bytes();
            element.x.self_to_montgomery_form();
            element.y.self_to_montgomery_form();
        }
    }
}

fn read_g2_elements_from_buffer(elements: &mut [G2Affine], buffer: &[u8]) {
    elements.copy_from_slice(&buffer);
    byteswap_g2(elements);
}

fn byteswap_g2(elements: &mut [G2Affine]) {
    let num_elements = elements.len();

    if cfg!(target_endian = "little") {
        for element in elements.iter_mut() {
            element.x.c0.data = element.x.c0.data.to_be_bytes();
            element.x.c1.data = element.x.c1.data.to_be_bytes();
            element.y.c0.data = element.y.c0.data.to_be_bytes();
            element.y.c1.data = element.y.c1.data.to_be_bytes();
            element.x.c0.self_to_montgomery_form();
            element.x.c1.self_to_montgomery_form();
            element.y.c0.self_to_montgomery_form();
            element.y.c1.self_to_montgomery_form();
        }
    }
}

fn get_file_size(filename: &str) -> std::io::Result<u64> {
    let metadata = std::fs::metadata(filename)?;
    Ok(metadata.len())
}

fn read_file_into_buffer(
    buffer: &mut [u8],
    size: usize,
    filename: &str,
    offset: u64,
    amount: usize,
) -> std::io::Result<()> {
    let mut file = File::open(filename)?;
    file.seek(SeekFrom::Start(offset))?;
    let actual_size = file.read(buffer)?;
    if actual_size != size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Only read {} bytes from file but expected {}.",
                actual_size, size
            ),
        ));
    }
    Ok(())
}

fn get_transcript_path(dir: &str, num: usize) -> String {
    format!("{}/monomial/transcript{:02}.dat", dir, num)
}

fn is_file_exist(file_name: &str) -> bool {
    Path::new(file_name).exists()
}

pub(crate) fn read_transcript_g1(
    monomials: &mut Vec<G1Affine>,
    degree: usize,
    dir: &str,
) -> Result<()> {
    let mut num = 0;
    let mut num_read = 0;
    let mut path = get_transcript_path(dir, num);

    while Path::new(&path).exists() && num_read < degree {
        let mut manifest = read_manifest(&path)?;

        let offset = std::mem::size_of::<Manifest>();
        let num_to_read = min(manifest.num_g1_points as usize, degree - num_read);
        let g1_buffer_size = std::mem::size_of::<Fq>() * 2 * num_to_read;

        let file = File::open(&path)?;
        let mut buffer = file.take(g1_buffer_size as u64);
        let mut monomial = &mut monomials[num_read..];

        // We must pass the size actually read to the second call, not the desired
        // g1_buffer_size as the file may have been smaller than this.
        buffer.read_exact(&mut monomial)?;

        byteswap_g1(&mut monomial);

        num_read += num_to_read;
        path = get_transcript_path(dir, num + 1);
    }

    if num_read < degree {
        return Err(anyhow!(
                "Only read {} points from {}, but require {}. Is your SRS large enough? \
                 Either run bootstrap.sh to download the transcript.dat files to `srs_db/ignition/`, \
                 or you might need to download extra transcript.dat files by editing \
                 `srs_db/download_ignition.sh` (but be careful, as this suggests you've \
                 just changed a circuit to exceed a new 'power of two' boundary).",
                num_read, path, degree
            )
        );
    }

    Ok(())
}

pub(crate) fn read_transcript_g2(g2_x: &mut [G2Affine], dir: &str) -> Result<()> {
    let g2_size = std::mem::size_of::<Fq2>() * 2;
    let mut path = format!("{}/g2.dat", dir);

    if Path::new(&path).exists() {
        let file = File::open(&path)?;
        let mut buffer = file.take(g2_size as u64);

        // Again, size passed to second function should be size actually read
        buffer.read_exact(g2_x)?;

        byteswap_g2(g2_x);

        return Ok(());
    }

    // Get transcript starting at g0.dat
    path = get_transcript_path(dir, 0);

    let mut manifest = read_manifest(&path)?;

    let g2_buffer_offset = std::mem::size_of::<Fq>() * 2 * manifest.num_g1_points as usize;
    let offset = std::mem::size_of::<Manifest>() + g2_buffer_offset;

    let mut file = File::open(&path)?;
    file.seek(SeekFrom::Start(offset as u64))?;
    let mut buf = vec![0; g2_size];
    file.read_exact(&mut buf[..]);

    g2_x.copy_from_slice(&mut buf);

    // Again, size passed to second function should be size actually read
    byteswap_g2(g2_x);

    Ok(())
}

fn read_transcript(
    monomials: &mut Vec<G1Affine>,
    g2_x: &mut G2Affine,
    degree: usize,
    path: &str,
) -> Result<()> {
    read_transcript_g1(monomials, degree, path)?;
    read_transcript_g2(g2_x, path)?;
    Ok(())
}
