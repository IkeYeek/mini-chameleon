import itertools
import re
import os
import sys
import subprocess

def main():
    csv_header = "variant;transA;transB;M;N;K;alpha;beta;gflops;success"
    csv_content = f"{csv_header}\n"
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    EXP_REGEX = re.compile(
        r"^tA=(?P<transA>NoTrans|Trans) tB=(?P<transB>NoTrans|Trans) M= +(?P<M>\d+) N= +(?P<N>\d+) K= +(?P<K>\d+) alpha= +(?P<alpha>[.e\-\+\d]+) beta= +(?P<beta>[.e\-\+\d]+): (?P<GFlops>[.e\-\+\d]+) GFlop\/s: (?P<result>SUCCESS|FAIL)$",
        flags=re.MULTILINE | re.DOTALL
    )
    COMPILE = True
    OUTPUT_FILE = os.path.abspath("dgemm.csv")
    BUILD_PATH = os.path.abspath("./build/debug")
    COMPILE_COMMAND = f"cmake --build {BUILD_PATH}"
    VERSION = ["seq", "seq_vec"]
    ITER = 1
    M = [10, 100, 1000]
    N = [10, 100, 1000]
    K = [10, 100, 1000]
    CHECK = True

    if not M:
        M = [100]
    if not K:
        K = [100]
    if not N:
        N = [100]

    if COMPILE:
        print("compiling...")
        compile_rez = subprocess.run(COMPILE_COMMAND, shell=True, capture_output=True)
        if compile_rez.returncode == 1:
            print(f"Compilation failed:\n\n{compile_rez.stderr.decode('utf-8')}", file=sys.stderr)
            exit(1)

    print("running experiments...")
    combs = [M, N, K, VERSION]
    combs_product = list(itertools.product(*combs))
    done = 0

    for comb in combs_product:
        m, n, k, v = comb
        size_args = f"-M {m} -N {n} -K {k}"
        check_arg = "--check" if CHECK else ""
        iter_arg = f"-i {ITER}"
        version_arg = f"-v {v}"
        bin_path = os.path.abspath("./build/debug/testings/perf_dgemm")
        cmdline = f"{bin_path} {version_arg} {size_args} {iter_arg} {check_arg}"
        print(f"running exp {done+1}/{len(combs_product)} ({cmdline})")
        process_rez = subprocess.run(cmdline, shell=True, capture_output=True, text=True)

        if process_rez.returncode == 1:
            print(f"Error during exp {done+1}, stopping here!\n\n{process_rez}", file=sys.stderr)
            exit(1)

        process_stdout = ansi_escape.sub('', process_rez.stdout)
        for exp_cap in EXP_REGEX.finditer(process_stdout):
            csv_content += ";".join([v, *exp_cap.groups()])
            csv_content += "\n"
        done += 1

    with open(OUTPUT_FILE, "w+") as output_file_fd:
        output_file_fd.write(csv_content)

    print(f"Done, content written in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
