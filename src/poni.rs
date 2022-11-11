pub struct Poni {
    pixel1: f32,
    pixel2: f32,
    distance: f32,
    poni1: f32,
    poni2: f32,
    rot1: f32,
    rot2: f32,
    rot3: f32,
    wavelength: f32,
}

impl Poni {
    pub fn new() -> Poni {
        Poni {
            pixel1: 0.0,
            pixel2: 0.0,
            distance: 0.0,
            poni1: 0.0,
            poni2: 0.0,
            rot1: 0.0,
            rot2: 0.0,
            rot3: 0.0,
            wavelength: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = Poni::new();
        assert_eq!(result.pixel1, 0.0);
    }
}

// Path: src\poni.rs
