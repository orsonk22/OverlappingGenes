import overlappingGenes as og
import traceback
import sys

try:
    print(f"Loaded og from {og.__file__}")
    prot1_len = 110
    prot2_len = 74
    overlap = 100
    print(f"Calling initial_seq_no_stops({prot1_len}, {prot2_len}, {overlap})")
    seq = og.initial_seq_no_stops(prot1_len, prot2_len, overlap)
    print("Success!")
    print(f"Seq length: {len(seq)}")
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")
    traceback.print_exc()
