use std::ops::{Index, IndexMut};
use std::mem;
use std::collections::{HashSet, HashMap};
use std::fs::File;
use std::error::Error;
use std::io::Write;
use std::fmt;

#[derive(Hash, Eq, PartialEq, Clone)]
struct LowerTriangularMatrix {
    data: Vec<bool>,    // Data stored in row-major order (only the entries that can possibly be non-zero)
    dim: usize
}

impl LowerTriangularMatrix {
    fn zeros(dim: usize) -> LowerTriangularMatrix {
        LowerTriangularMatrix {
            data: vec![false; dim * (dim + 1) / 2],
            dim,
        }
    }
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
            for _ in 0 .. self.dim - i {
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

fn canonical_matrix(dim: usize, rank: usize, kind: usize) -> LowerTriangularMatrix {
    assert!(rank <= dim);
    assert!(kind <= 1);
    assert!((rank > 0 && rank % 2 == 0) || kind == 0);
    
    let mut output = LowerTriangularMatrix::zeros(dim);
    let mut i = 0;    
    if rank % 2 == 1 {
        output[[0, 0]] = true;
        i += 1;
    } else if kind == 1 {
        output[[0, 0]] = true;
        output[[1, 0]] = true;
        output[[1, 1]] = true;
        i += 2;
    }
    while i < rank {
        output[[i + 1, i]] = true;
        i += 2;
    }
    output
}


struct CanonicalMatrix {
    mat: LowerTriangularMatrix,
    rank: usize,
    kind: usize,
}

fn all_canonical_matrices(dim: usize) -> Vec<CanonicalMatrix> {
    let mut out: Vec<CanonicalMatrix> = Vec::new();
    for rank in 0..(dim + 1) {
        out.push(CanonicalMatrix {
            mat: canonical_matrix(dim, rank, 0),
            rank,
            kind: 0
        });
        if rank % 2 == 0 && rank > 0 {
            out.push(CanonicalMatrix {
                mat: canonical_matrix(dim, rank, 1),
                rank,
                kind: 1
            });
        }
    }   
    out    
}

fn compute_orbit(mat: LowerTriangularMatrix) -> HashSet<LowerTriangularMatrix> {
    let mut full_set: HashSet<LowerTriangularMatrix> = HashSet::new();
    let mut new_set: HashSet<LowerTriangularMatrix> = HashSet::new();
    new_set.insert(mat);
    while !new_set.is_empty() {
        let mut next_set: HashSet<LowerTriangularMatrix> = HashSet::new();
        for mat in new_set {
            let new_mat = generator_cycle(&mat);
            if full_set.insert(new_mat.clone()) {
                next_set.insert(new_mat);
            }

            let new_mat = generator_transvection(&mat);
            if full_set.insert(new_mat.clone()) {
                next_set.insert(new_mat);
            }
        }
        new_set = next_set;
    }
    full_set
}

fn create_full_space_rec(more_dim: usize, prefix: &mut Vec<bool>, out: &mut Vec<Vec<bool>>) {
    if more_dim == 0 {
        out.push(prefix.clone());
    } else {
        prefix.push(false);
        create_full_space_rec(more_dim - 1, prefix, out);
        prefix.pop();

        prefix.push(true);
        create_full_space_rec(more_dim - 1, prefix, out);
        prefix.pop();
    }
}

fn create_full_space(dim: usize) -> Vec<Vec<bool>> {
    let mut out: Vec<Vec<bool>> = Vec::new();
    let mut prefix: Vec<bool> = Vec::new();
    create_full_space_rec(dim, &mut prefix, &mut out);
    out
}

fn evaluate(mat: &LowerTriangularMatrix, v: &[bool]) -> bool {
    let mut out = false;
    for i in 0..mat.dim {
        for j in 0..i + 1 {
            out ^= mat[[i, j]] && v[i] && v[j];
        }
    }
    out
}

fn evaluate_all(mat: &LowerTriangularMatrix) -> Vec<bool> {
    create_full_space(mat.dim).iter().map(|v| evaluate(mat, v)).collect()
}

fn compute_weight(mat: &LowerTriangularMatrix) -> usize {
    evaluate_all(mat).iter().map(|b| *b as usize).sum()
}

fn compute_intersection_numbers(mat: &LowerTriangularMatrix, orbit: &HashSet<LowerTriangularMatrix>) -> Vec<usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let v1 = evaluate_all(mat);
    for m in orbit {
        let v2 = evaluate_all(m);
        if v2 == v1 { 
            continue; 
        }
        let mut cnt = 0;
        for (x, y) in v1.iter().zip(&v2) {
            if *x && *y {
                cnt += 1;
            }
        }
        *counts.entry(cnt).or_insert(0) += 1;
    }

    let mut counts_vec: Vec<usize> = Vec::new();
    for i in 0 ..= v1.len() {
        counts_vec.push(*counts.get(&i).unwrap_or(&0));
    }
    counts_vec
}

fn write_design(cm: &CanonicalMatrix) -> Result<(), Box<dyn Error>> {
    let orbit = compute_orbit(cm.mat.clone());
    let kind = if cm.rank % 2 == 0 { if cm.kind == 1 { "B" } else { "A" } } else { "" };
    let filename = format!("output/design_l{}_r{}{}_w{}_s{}.json", 
        2usize.pow(cm.mat.dim as u32), cm.rank, kind, compute_weight(&cm.mat), orbit.len());
    println!("Writing {}", filename);
    let mut file = File::create(filename)?;
    writeln!(file, "[")?;
    for (i, m) in orbit.iter().enumerate() {
        let vec = evaluate_all(m);
        write!(file, "[")?;
        for (j, x) in vec.iter().enumerate() {
            if *x {
                write!(file, "1")?;
            } else {
                write!(file, "0")?;
            }
            if j != vec.len() - 1 {
                write!(file, ",")?;
            }
        }
        write!(file, "]")?;
        if i != orbit.len() - 1 {
            writeln!(file, ",")?;
        }
    }
    writeln!(file, "\n]")?;
    file.sync_all()?;
    Ok(())
}

fn main() {
    // let dim = 8;
    for dim in 2 ..= 8 {
        let canonical_matrices = all_canonical_matrices(dim);

        for cm in canonical_matrices {
            if cm.rank == 0 {
                continue;
            }
            write_design(&cm).unwrap();
            // let set = compute_orbit(cm.mat.clone());
            // println!("dim {}, rank {}, kind {}, weight {}, orbit size {}", dim, cm.rank, cm.kind, compute_weight(&cm.mat), set.len());
            // println!("intersections: {:?}", compute_intersection_numbers(&cm.mat, &set));
        }   
    
    }
}
