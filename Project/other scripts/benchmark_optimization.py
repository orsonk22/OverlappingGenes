
import time
import numpy as np
import overlappingGenes as og
import os

def run_benchmark():
    print("Setting up benchmark...")
    
    # Mock data generation
    L_aa = 50
    L_nuc = L_aa * 3
    
    h_size = 21 * L_aa
    J_size = int((L_aa * (L_aa - 1) / 2) * 441) + 441 # Safety buffer
    
    print(f"Generating mock parameters (L={L_aa} aa)...")
    hvec1 = np.random.randn(h_size).astype(np.float32)
    Jvec1 = np.random.randn(J_size).astype(np.float32)
    
    hvec2 = np.random.randn(h_size).astype(np.float32)
    Jvec2 = np.random.randn(J_size).astype(np.float32)
    
    DCA_params_1 = [Jvec1, hvec1]
    DCA_params_2 = [Jvec2, hvec2]
    
    overlapLen = 20
    print("Generating valid initial sequence...")
    initial_seq = og.initial_seq_no_stops(L_aa, L_aa, overlapLen, quiet=True)
    
    print("Warming up JIT...")
    # Warmup
    og.overlapped_sequence_generator_int(
        DCA_params_1, DCA_params_2, 
        initial_seq, 
        T1=1.0, T2=1.0, 
        numberofiterations=100, 
        quiet=True, 
        whentosave=0.5
    )
    
    print("Starting benchmark (100000 iterations)...")
    start_time = time.time()
    
    n_iter = 100000
    og.overlapped_sequence_generator_int(
        DCA_params_1, DCA_params_2, 
        initial_seq, 
        T1=1.0, T2=1.0, 
        numberofiterations=n_iter, 
        quiet=True, 
        whentosave=0.5
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Benchmark completed in {duration:.4f} seconds")
    print(f"Iterations per second: {n_iter / duration:.2f}")

if __name__ == "__main__":
    run_benchmark()
