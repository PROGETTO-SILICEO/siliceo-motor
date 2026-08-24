//! Sampling delle distribuzioni di logits — F2.
//!
//! Semantica allineata a llama.cpp / HF transformers:
//! - temperature 0 (o `greedy`) → argmax puro
//! - temperature > 0 → scala dei logits, filtro top-k, filtro nucleus top-p,
//!   softmax sul residuo, campionamento multinomiale.
//!
//! RNG interno (splitmix64) seedabile: stesso seed ⇒ stessa sequenza,
//! così i test sono riproducibili e il parity è verificabile.

/// RNG splitmix64 — piccolo, veloce, deterministico, senza dipendenze.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }

    /// Default: seed dal tempo (per uso interattivo).
    pub fn from_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self::new(nanos)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniforme in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // 53 bit di mantissa per una buona uniformità
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    /// Temperatura (0 = greedy/disabled).
    pub temperature: f32,
    /// Top-k: conserva solo le k probabilità più alte (None = disattivo).
    pub top_k: Option<usize>,
    /// Top-p (nucleus): conserva il minor numero di token con massa cumulata >= p.
    pub top_p: Option<f32>,
}

impl Sampler {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: None, top_p: None }
    }

    pub fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Self {
        Self { temperature, top_k, top_p }
    }

    /// Campiona un token dai logits. Ritorna l'indice del token scelto.
    ///
    /// Nota tie-breaking: su valori identici vince l'indice PIÙ BASSO
    /// (stessa convenzione dell'argmax usato nel debug F1).
    pub fn sample(&mut self, logits: &[f32], rng: &mut Rng) -> usize {
        // Greedy: argmax diretto.
        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        // 1. scala per temperatura
        let mut probs: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();

        // 2. softmax globale (numerically stable)
        softmax(&mut probs);

        // 3. ordina gli indici per probabilità decrescente (tie: indice basso prima)
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });

        // 4. top-k
        if let Some(k) = self.top_k {
            let k = k.min(idx.len()).max(1);
            idx.truncate(k);
        }

        // 5. top-p (nucleus): conserva finché la massa cumulata non raggiunge p;
        //    il token che attraversa la soglia viene INCLUSO (convenzione HF).
        if let Some(p) = self.top_p {
            let mut cum = 0.0f32;
            let mut cut = idx.len();
            for (i, &j) in idx.iter().enumerate() {
                cum += probs[j];
                if cum >= p {
                    cut = i + 1;
                    break;
                }
            }
            idx.truncate(cut.max(1));
        }

        // 6. multinomiale sul residuo
        let total: f32 = idx.iter().map(|&j| probs[j]).sum();
        let mut r = rng.next_f32() * total;
        for &j in &idx {
            r -= probs[j];
            if r <= 0.0 {
                return j;
            }
        }
        // fallback numerico (r leggermente > totale per arrotondamento)
        *idx.last().unwrap()
    }
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

fn softmax(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sceglie_argmax() {
        let mut s = Sampler::greedy();
        let mut rng = Rng::new(1);
        assert_eq!(s.sample(&[0.1, 5.0, 3.0], &mut rng), 1);
        // tie: vince l'indice più basso
        assert_eq!(s.sample(&[2.0, 2.0, 1.0], &mut rng), 0);
    }

    #[test]
    fn temperatura_zero_e_greedy_equivalenti() {
        let mut s = Sampler::new(0.0, Some(40), Some(0.9));
        let mut rng = Rng::new(42);
        assert_eq!(s.sample(&[-1.0, 9.0, 2.0, 9.0], &mut rng), 1);
    }

    #[test]
    fn topk_1_è_argmax_con_qualsiasi_temperatura() {
        let mut s = Sampler::new(1.5, Some(1), None);
        let mut rng = Rng::new(7);
        assert_eq!(s.sample(&[0.2, 0.1, 8.8], &mut rng), 2);
    }

    #[test]
    fn seed_stesso_distribuzione_stessa() {
        let logits = vec![0.1f32; 100];
        let mut a = Sampler::new(0.8, None, None);
        let mut b = Sampler::new(0.8, None, None);
        let mut ra = Rng::new(123);
        let mut rb = Rng::new(123);
        for _ in 0..50 {
            assert_eq!(a.sample(&logits, &mut ra), b.sample(&logits, &mut rb));
        }
    }

    #[test]
    fn topp_piccolo_cola_al_top() {
        // distribuzione molto concentrata: top-p=0.01 deve scegliere il massimo
        let mut logits = vec![0.0f32; 10];
        logits[3] = 20.0; // dominante schiacciante
        let mut s = Sampler::new(1.0, None, Some(0.01));
        let mut rng = Rng::new(99);
        assert_eq!(s.sample(&logits, &mut rng), 3);
    }

    #[test]
    fn rng_uniformo_in_range() {
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
