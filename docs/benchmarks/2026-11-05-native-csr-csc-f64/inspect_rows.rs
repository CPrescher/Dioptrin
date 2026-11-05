#![allow(dead_code)]
trait Pixel: Copy {fn to_f64(self)->f64;}
impl Pixel for f64 {#[inline(always)] fn to_f64(self)->f64 {self}}
trait PixelIndex: Copy {fn pixel(self)->usize;}
impl PixelIndex for u32 {#[inline(always)] fn pixel(self)->usize {self as usize}}
#[inline(always)] fn madd(a:f64,b:f64,c:f64)->f64 {a.mul_add(b,c)}
#[inline(always)]
fn row<
    I: PixelIndex,
    P: Pixel,
    const FAST: bool,
    const SIGNAL: bool,
    const NORM: bool,
    const VAR: bool,
    const COUNT: bool,
    const WEIGHT: bool,
    const MASK: bool,
>(
    start: usize,
    end: usize,
    cols: &[I],
    weights: &[f64],
    mask: &[u8],
    signal: &[P],
    norm: &[f64],
) -> [f64; 4] {
    let mut s = [0.0; 4];
    let mut n = [0.0; 4];
    let mut v = [0.0; 4];
    let mut c = [0.0; 4];
    let mut k = start;
    // SAFETY: these tables are private and constructed only from valid LUTs.
    // Integrator validates image/mask dimensions; const branches avoid reads of
    // absent arrays. Compact conversion checks every index before narrowing.
    unsafe {
        macro_rules! visit {
            ($lane:expr, $at:expr) => {{
                let lane = $lane;
                let j = cols.get_unchecked($at).pixel();
                if !MASK || *mask.get_unchecked(j) == 0 {
                    let w = if WEIGHT {
                        *weights.get_unchecked($at)
                    } else {
                        1.0
                    };
                    if SIGNAL {
                        let x = signal.get_unchecked(j).to_f64();
                        if WEIGHT {
                            s[lane] = if FAST {
                                madd(w, x, s[lane])
                            } else {
                                s[lane] + w * x
                            };
                            if VAR {
                                let variance = x.max(0.0);
                                v[lane] = if FAST {
                                    madd(w * w, variance, v[lane])
                                } else {
                                    v[lane] + w * w * variance
                                };
                            }
                        } else {
                            s[lane] += x;
                            if VAR {
                                v[lane] += x.max(0.0);
                            }
                        }
                    }
                    if NORM {
                        let x = *norm.get_unchecked(j);
                        n[lane] = if WEIGHT {
                            if FAST {
                                madd(w, x, n[lane])
                            } else {
                                n[lane] + w * x
                            }
                        } else {
                            n[lane] + x
                        };
                    }
                    if COUNT {
                        c[lane] += w;
                    }
                }
            }};
        }
        if FAST {
            while k + 4 <= end {
                for lane in 0..4 {
                    visit!(lane, k + lane);
                }
                k += 4;
            }
            for lane in 0..end - k {
                visit!(lane, k + lane);
            }
        } else {
            while k < end {
                visit!(0, k);
                k += 1;
            }
        }
    }
    let reduce = |x: [f64; 4]| {
        if FAST {
            (x[0] + x[1]) + (x[2] + x[3])
        } else {
            x[0]
        }
    };
    [reduce(s), reduce(n), reduce(v), reduce(c)]
}


#[unsafe(no_mangle)]
pub fn masked(cols:&[u32],weights:&[f64],mask:&[u8],signal:&[f64])->f64 {
 row::<u32,f64,true,true,false,false,false,true,true>(0,cols.len(),cols,weights,mask,signal,&[])[0]
}
#[unsafe(no_mangle)]
pub fn unmasked(cols:&[u32],weights:&[f64],signal:&[f64])->f64 {
 row::<u32,f64,true,true,false,false,false,true,false>(0,cols.len(),cols,weights,&[],signal,&[])[0]
}
