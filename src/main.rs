use std::fmt;
use std::ops::{Index, IndexMut};
use std::mem;

#[derive(Hash, Eq, PartialEq, Clone)]
struct LowerTriangularMatrix {
    data: Vec<bool>,    // Data stored in row-major order (only the entries that can possibly be non-zero)
    dim: usize
}

impl Index<[usize; 2]> for LowerTriangularMatrix {
    type Output = bool;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let row = index[0];
        let col = index[1];
        assert!(col <= row);
        &self.data[row * (row + 1) / 2 + col]
    }
}

impl IndexMut<[usize; 2]> for LowerTriangularMatrix {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let row = index[0];
        let col = index[1];
        assert!(col <= row);
        &mut self.data[row * (row + 1) / 2 + col]
    }
}

impl fmt::Display for LowerTriangularMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.dim {
            for j in 0 .. i + 1 {
                if self[[i, j]] {
                    write!(f, "1 ")?;
                } else {
                    write!(f, "0 ")?;
                }
            }
            for j in 0 .. self.dim - i {
                write!(f, "  ")?;
            }
            
            writeln!(f, "")?;
        }
        Ok(())
    }
}

fn generator_cycle(input: &LowerTriangularMatrix) -> LowerTriangularMatrix {
    let mut output = LowerTriangularMatrix {
        data: vec![false; input.data.len()],
        dim: input.dim,
    };

    for row in 0..input.dim {
        for col in 0 .. row + 1 {
            let mut row1 = (row + 1) % input.dim;
            let mut col1 = (col + 1) % input.dim;
            if col1 > row1 {
                mem::swap(&mut row1, &mut col1);
            } 
            output[[row1, col1]] = input[[row, col]]
        }
    }
    output
}

fn generator_transvection(input: &LowerTriangularMatrix) -> LowerTriangularMatrix {
    let mut output = LowerTriangularMatrix {
        data: input.data.clone(),
        dim: input.dim,
    };

    output[[1, 1]] ^= input[[0, 0]];
    for row in 1..input.dim {
        output[[row, 1]] ^= input[[row, 0]];
    }
    return output;
}

fn main() {
    let mut mat = LowerTriangularMatrix {
        data: vec![false; 10],
        dim: 4
    };
    mat.data[9] = true;
    // let mat1 = generator_cycle(&mat);
    let mat1 = generator_transvection(&mat);
    println!("{}", mat);
    println!("{}", mat1);
}
