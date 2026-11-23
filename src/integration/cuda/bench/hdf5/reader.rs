use super::Result;
use libloading::Library;
use std::{
    ffi::{CString, c_char, c_void},
    marker::PhantomData,
    path::Path,
};

// HDF5 1.14/2.0 public C ABI. This reader is used only by the benchmark.
type Hid = i64;
type Hsize = u64;
type Close = unsafe extern "C" fn(Hid) -> i32;
const DEFAULT: Hid = 0;

pub(super) struct LibraryApi {
    _lib: Library,
    close_library: unsafe extern "C" fn() -> i32,
    native_u32: Hid,
    open_file: unsafe extern "C" fn(*const c_char, u32, Hid) -> Hid,
    close_file: Close,
    open_dataset: unsafe extern "C" fn(Hid, *const c_char, Hid) -> Hid,
    close_dataset: Close,
    get_space: unsafe extern "C" fn(Hid) -> Hid,
    rank: unsafe extern "C" fn(Hid) -> i32,
    dims: unsafe extern "C" fn(Hid, *mut Hsize, *mut Hsize) -> i32,
    create_space: unsafe extern "C" fn(i32, *const Hsize, *const Hsize) -> Hid,
    close_space: Close,
    select: unsafe extern "C" fn(
        Hid,
        i32,
        *const Hsize,
        *const Hsize,
        *const Hsize,
        *const Hsize,
    ) -> i32,
    read: unsafe extern "C" fn(Hid, Hid, Hid, Hid, Hid, *mut c_void) -> i32,
}

struct Handle<'a> {
    id: Hid,
    close: Close,
    _api: PhantomData<&'a LibraryApi>,
}

impl Drop for Handle<'_> {
    fn drop(&mut self) {
        // SAFETY: Each nonnegative handle is owned by exactly one wrapper.
        unsafe { (self.close)(self.id) };
    }
}

impl LibraryApi {
    pub(super) fn load(path: &Path) -> Result<Self> {
        // SAFETY: The harness passes the path of the installed native HDF5
        // library. Symbols below have their public HDF5 signatures.
        unsafe {
            let lib = Library::new(path)?;
            let open = lib.get::<unsafe extern "C" fn() -> i32>(b"H5open")?;
            if open() < 0 {
                return Err("H5open failed".into());
            }
            Ok(Self {
                close_library: *lib.get(b"H5close")?,
                native_u32: **lib.get::<*const Hid>(b"H5T_NATIVE_UINT32_g")?,
                open_file: *lib.get(b"H5Fopen")?,
                close_file: *lib.get(b"H5Fclose")?,
                open_dataset: *lib.get(b"H5Dopen2")?,
                close_dataset: *lib.get(b"H5Dclose")?,
                get_space: *lib.get(b"H5Dget_space")?,
                rank: *lib.get(b"H5Sget_simple_extent_ndims")?,
                dims: *lib.get(b"H5Sget_simple_extent_dims")?,
                create_space: *lib.get(b"H5Screate_simple")?,
                close_space: *lib.get(b"H5Sclose")?,
                select: *lib.get(b"H5Sselect_hyperslab")?,
                read: *lib.get(b"H5Dread")?,
                _lib: lib,
            })
        }
    }

    fn handle(&self, id: Hid, close: Close, error: &'static str) -> Result<Handle<'_>> {
        if id < 0 {
            return Err(error.into());
        }
        Ok(Handle {
            id,
            close,
            _api: PhantomData,
        })
    }

    pub(super) fn dataset(&self, path: &str, name: &str) -> Result<Dataset<'_>> {
        let path = CString::new(path)?;
        let name = CString::new(name)?;
        // SAFETY: Path and dataset strings are nul-terminated. The file is
        // opened read-only and all identifiers stay live through each call.
        let file = self.handle(
            unsafe { (self.open_file)(path.as_ptr(), 0, DEFAULT) },
            self.close_file,
            "cannot open Eiger HDF5 file",
        )?;
        let dataset = self.handle(
            unsafe { (self.open_dataset)(file.id, name.as_ptr(), DEFAULT) },
            self.close_dataset,
            "cannot find Eiger dataset",
        )?;
        let space = self.handle(
            unsafe { (self.get_space)(dataset.id) },
            self.close_space,
            "cannot obtain Eiger dataspace",
        )?;
        let mut dims = [0; 3];
        if unsafe { (self.rank)(space.id) } != 3
            || unsafe { (self.dims)(space.id, dims.as_mut_ptr(), std::ptr::null_mut()) } != 3
            || dims.contains(&0)
        {
            return Err("expected nonempty frames × rows × columns in Eiger HDF5".into());
        }
        let dims = [
            usize::try_from(dims[0])?,
            usize::try_from(dims[1])?,
            usize::try_from(dims[2])?,
        ];
        dims[1]
            .checked_mul(dims[2])
            .ok_or("HDF5 frame is too large")?;
        Ok(Dataset {
            api: self,
            space,
            dataset,
            _file: file,
            dims,
        })
    }
}

impl Drop for LibraryApi {
    fn drop(&mut self) {
        // SAFETY: Dataset and dataspace handles borrow the library wrapper.
        unsafe { (self.close_library)() };
    }
}

pub(super) struct Dataset<'a> {
    api: &'a LibraryApi,
    space: Handle<'a>,
    dataset: Handle<'a>,
    _file: Handle<'a>,
    pub(super) dims: [usize; 3],
}

impl Dataset<'_> {
    pub(super) fn read(&mut self, first: usize, destination: &mut [u32]) -> Result<()> {
        let [frames, rows, columns] = self.dims;
        let pixels = rows * columns;
        let count = destination.len() / pixels;
        if count == 0
            || !destination.len().is_multiple_of(pixels)
            || first > frames
            || count > frames - first
        {
            return Err("HDF5 read must cover complete frames inside the dataset".into());
        }
        let start = [first as Hsize, 0, 0];
        let extent = [count as Hsize, rows as Hsize, columns as Hsize];
        let null = std::ptr::null();
        let api = self.api;
        // SAFETY: The file selection and memory extent contain exactly the
        // number of u32 values in destination. All pointers remain live.
        unsafe {
            // Match the file selection's rank and shape so HDF5 can place
            // each decoded frame directly in the destination.
            let memory = api.handle(
                (api.create_space)(3, extent.as_ptr(), null),
                api.close_space,
                "cannot create memory dataspace",
            )?;
            if (api.select)(
                self.space.id,
                0,
                start.as_ptr(),
                null,
                extent.as_ptr(),
                null,
            ) < 0
                || (api.read)(
                    self.dataset.id,
                    api.native_u32,
                    memory.id,
                    self.space.id,
                    DEFAULT,
                    destination.as_mut_ptr().cast(),
                ) < 0
            {
                return Err(format!(
                    "could not decode HDF5 frames {first} through {}",
                    first + count - 1
                )
                .into());
            }
        }
        Ok(())
    }
}
