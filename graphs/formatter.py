import re
import argparse
import os

def parse_line(line):
    match = re.search(r'M=\s*(\d+).*?N=\s*(\d+).*?K=\s*(\d+).*?:\s*([\d.+-eE]+)\s*GFlop/s', line)
    
    if match:
        M = int(match.group(1))
        N = int(match.group(2))
        K = int(match.group(3))
        gflops = float(match.group(4))
        return M, N, K, gflops
    
    return None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Output parser',
                    description='Parses benchmark output files for M, N, K, and GFlops.',
                    epilog='Text at the bottom of help')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('-i', '--input', type=str, help='Input file name')
    parser.add_argument('-l', '--label', type=str, help='Custom label for the data')
    parser.add_argument('-al', '--auto-label', action='store_true', help='Automatically generate a label from the input file name')

    args = parser.parse_args()

    if args.input == None:
        print("Error: Input file must be specified with -i or --input")
        parser.print_usage()
        exit(1)

    active_label = (args.label != None) or (args.auto_label == True)
    if args.auto_label == True:
        parts = args.input.split('-')
        if len(parts) > 1:
            args.label = parts[1] + ("_" + parts[-1].split(".")[0] if len(parts) >= 4 else "")
        else:
            args.label = args.input

    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)


    if args.output:
        output_name = args.output
    else:
        parts = args.input.split('-')
        if len(parts) >= 3:
            output_name = parts[0] + "_" + parts[2] + ".out"
        else:
            output_name = args.input + ".out"

    header = "M N K GFlops Label" if active_label else "M N K GFlops"
    
    output_file_exists = os.path.exists(output_name)

    if not output_file_exists:
        output_lines = [header]
    else:
        output_lines = []

    for line in lines:
        stripped_line = line.strip()
        M, N, K, gflops = parse_line(stripped_line)
        
        if gflops is not None:
            label_str = f" {args.label}" if active_label else ""
            output_lines.append(f"{M} {N} {K} {gflops:.6f}{label_str}")
        else:
            if stripped_line:
                print("Skipped line (no match):", stripped_line)

    """else:
        parts = args.input.split('-')
        if len(parts) >= 3:
            output_name = parts[0] + "_" + parts[2] + ".out"
        else:
            output_name = args.input + ".out"""
    
    write_mode = 'a' if output_file_exists else 'w'
    try:
        with open(output_name, write_mode) as f:              
            f.write("\n".join(output_lines))
            f.write("\n")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        exit(1)

    print(f"All cleaned data written to {output_name}")