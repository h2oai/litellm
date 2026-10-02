#!/usr/bin/env python3
"""
Log filter to remove API keys from litellm output.
Filters out lines containing ?key= to prevent API key leakage.
"""
import sys
import subprocess
import threading
import re

def filter_log_line(line):
    """Filter out API keys from log lines."""
    # Pattern to match lines with ?key= parameter
    if '?key=' in line:
        # Replace the key value with [REDACTED]
        filtered_line = re.sub(r'\?key=[^&\s"\']*', '?key=[REDACTED]', line)
        return filtered_line
    return line

def filter_stream(input_stream, output_stream):
    """Filter a stream line by line."""
    try:
        for line in iter(input_stream.readline, b''):
            if line:
                decoded_line = line.decode('utf-8', errors='replace')
                filtered_line = filter_log_line(decoded_line)
                output_stream.write(filtered_line.encode('utf-8'))
                output_stream.flush()
    except Exception as e:
        print(f"Filter error: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_api_keys.py <command> [args...]", file=sys.stderr)
        sys.exit(1)
    
    # Start the subprocess
    process = subprocess.Popen(
        sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0  # Unbuffered
    )
    
    # Filter stdout and stderr in separate threads
    stdout_thread = threading.Thread(
        target=filter_stream,
        args=(process.stdout, sys.stdout.buffer)
    )
    stderr_thread = threading.Thread(
        target=filter_stream, 
        args=(process.stderr, sys.stderr.buffer)
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process to complete
    process.wait()
    
    # Wait for filter threads to complete
    stdout_thread.join()
    stderr_thread.join()
    
    sys.exit(process.returncode)

if __name__ == "__main__":
    main()